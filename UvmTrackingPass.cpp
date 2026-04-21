#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Constants.h"
#include <llvm-20/llvm/Support/raw_ostream.h>
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

static constexpr uint64_t PAGE_SHIFT = 12;
static constexpr uint64_t PAGE_SIZE  = (1ULL << PAGE_SHIFT); // 4096

// Trace through addrspacecast and GEPs to find the true allocation address space.
// In LLVM 20 opaque-pointer mode, __shared__ arrays are cast to generic AS 0 via
// an inline ConstantExpr addrspacecast (ptr addrspace(3) @As to ptr), which appears
// as the base pointer of GEPs but is NOT an AddrSpaceCastInst — it's a ConstantExpr.
static unsigned trueAddressSpace(llvm::Value *V) {
    while (true) {
        if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(V))
            return ASC->getPointerOperand()->getType()->getPointerAddressSpace();
        // ConstantExpr addrspacecast — the common case for __shared__ globals
        if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
            if (CE->getOpcode() == llvm::Instruction::AddrSpaceCast)
                return CE->getOperand(0)->getType()->getPointerAddressSpace();
            if (CE->getOpcode() == llvm::Instruction::GetElementPtr) {
                V = CE->getOperand(0);
                continue;
            }
        }
        if (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(V)) {
            V = GEP->getPointerOperand();
            continue;
        }
        break;
    }
    return V->getType()->getPointerAddressSpace();
}


llvm::Function *MarkFuncPtr = nullptr;


enum class PageInvariance {
    Invariant,          // provably touches same page every iteration
    Variant,            // provably touches different pages
    MaybeInvariant,     // stride*N < PAGE_SIZE but base alignment unknown
    Unknown             // SCEV couldn't model it
};

struct PageInvarianceResult {
    PageInvariance kind;
    const llvm::SCEV *stride  = nullptr;  // for diagnostics
    uint64_t         tripCount = 0;
    uint64_t         byteRange = 0;       // stride * (N-1), if known
};

PageInvarianceResult checkPageInvariance(
        llvm::Value       *Ptr,
        llvm::Loop        *L,
        llvm::ScalarEvolution &SE) {

    const llvm::SCEV *PtrSCEV = SE.getSCEV(Ptr);

    // ── Case 1: Fully loop-invariant address → trivially page-invariant ──
    if (SE.isLoopInvariant(PtrSCEV, L)){
        llvm::errs() << "[UvmPass] Loop-invariant pointer detected at" << *PtrSCEV
                   << "\n";
        return {PageInvariance::Invariant};
    }

    // ── Case 2: Affine recurrence {base, +, stride}_L ──
    auto *AR = llvm::dyn_cast<llvm::SCEVAddRecExpr>(PtrSCEV);
    if (!AR || AR->getLoop() != L || !AR->isAffine()){
        //llvm::errs() << "[UvmPass] Non-affine or non-additive SCEV for pointer: " << *PtrSCEV << "\n";
        return { PageInvariance::Unknown };
    }

    const llvm::SCEV *StrideSCEV = AR->getStepRecurrence(SE);
    const llvm::SCEV *BaseSCEV   = AR->getStart();

    // Get trip count (number of iterations, not backedge count)
    uint64_t TC = SE.getSmallConstantTripCount(L);
    
    llvm::errs() << "[UvmPass] Analyzing affine recurrence: " << *AR << "\n";

    // ── Case 2a: Stride is 0 → pointer never moves ──
    if (StrideSCEV->isZero()){
        llvm::errs() << "[UvmPass] Zero stride detected: " << *StrideSCEV << "\n";
        return { PageInvariance::Invariant, StrideSCEV, TC, 0 };
    }

    // ── Case 2b: Constant stride and known trip count ──
    if (auto *ConstStride = llvm::dyn_cast<llvm::SCEVConstant>(StrideSCEV)) {
        llvm::errs() << "[UvmPass] Constant stride detected: " << *ConstStride << "\n";
        int64_t S = ConstStride->getAPInt().getSExtValue();
        uint64_t absStride = std::abs(S);

        if (TC > 0) {
            uint64_t totalRange = absStride * (TC - 1);
            // Sufficient condition: total byte range < PAGE_SIZE
            // This guarantees no page crossing regardless of base alignment
            if (totalRange < PAGE_SIZE)
                return { PageInvariance::MaybeInvariant, StrideSCEV, TC, totalRange };
            //         ↑ "Maybe" because base could straddle a page boundary.
            //           E.g., range=100 bytes but base=0xFFC → crosses 0x1000.
            //           Emit a runtime check for the base (see below).

            // If stride is a multiple of PAGE_SIZE, every iteration hits a
            // *different* page — variant but regular (useful for prefetch analysis)
            if (absStride % PAGE_SIZE == 0)
                return { PageInvariance::Variant, StrideSCEV, TC, totalRange };

            // totalRange >= PAGE_SIZE and stride not page-aligned → variant
            return { PageInvariance::Variant, StrideSCEV, TC, totalRange };
        }

        // TC unknown but stride known: try page-fit via SCEV range first,
        // then fall back to Variant so the pass can try runtime batching.
        auto Range = SE.getUnsignedRange(AR);
        if (!Range.isFullSet()) {
            uint64_t SCEVRange = (Range.getUpper() - Range.getLower())
                                     .getLimitedValue();
            if (SCEVRange < PAGE_SIZE)
                return { PageInvariance::MaybeInvariant, StrideSCEV, 0, SCEVRange };
        }
        // Constant stride, TC unknown, large range → Variant; emitBatchMarkAccess
        // will resolve TC at compile time via getBackedgeTakenCount.
        return { PageInvariance::Variant, StrideSCEV, 0, 0 };
    }

    // ── Case 2c: Non-constant stride, affine recurrence ──────────────────────
    // Stride is loop-invariant (guaranteed by isAffine() above).  Return Variant
    // so emitBatchMarkAccess can expand both stride and TC via SCEVExpander.
    // If SCEVExpander can't resolve them at compile time it will bail out safely.
    llvm::errs() << "[UvmPass] Non-constant affine stride, returning Variant: " << *PtrSCEV << "\n";
    return { PageInvariance::Variant, StrideSCEV, TC, 0 };
}


bool isPageAlignedStride(const llvm::SCEV *StrideSCEV) {
    // Stride must be a compile-time constant for us to reason about it
    auto *ConstStride = llvm::dyn_cast<llvm::SCEVConstant>(StrideSCEV);
    if (!ConstStride) return false;

    int64_t S = ConstStride->getAPInt().getSExtValue();
    if (S == 0) return false;

    return (std::abs(S) % PAGE_SIZE) == 0;
}


void hoistMarkPage(llvm::Value *Ptr, llvm::Loop *L,
                   llvm::Function *MarkPageFn) {
    // Get the preheader — the block that jumps into the loop
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return; // can't hoist without a single preheader

    // Insert before the preheader's terminator (the branch into the loop)
    llvm::IRBuilder<> B(Preheader->getTerminator());

    // Cast ptr to i64, shift right by PAGE_SHIFT to get page number
    auto *I64Ty  = llvm::Type::getInt64Ty(Preheader->getContext());
    auto *PtrInt = B.CreatePtrToInt(Ptr, I64Ty, "ptr.int");
    auto *PageNo = B.CreateLShr(PtrInt,
                       llvm::ConstantInt::get(I64Ty, PAGE_SHIFT), "page.no");

    // Call your existing mark_page instrumentation function
    B.CreateCall(MarkPageFn, {PageNo});
}

void hoistMarkPageConditional(llvm::Value *Ptr, llvm::Loop *L,
                               uint64_t ByteRange,
                               llvm::Function *MarkPageFn) {
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return;

    llvm::LLVMContext &Ctx = Preheader->getContext();
    llvm::Function   *F   = Preheader->getParent();
    auto *I64Ty = llvm::Type::getInt64Ty(Ctx);

    // ── 1. Emit the runtime check at end of preheader ──
    llvm::IRBuilder<> B(Preheader->getTerminator());

    llvm::Value *PtrInt  = B.CreatePtrToInt(Ptr, I64Ty);
    llvm::Value *PageOff = B.CreateAnd(PtrInt,
                               llvm::ConstantInt::get(I64Ty, PAGE_SIZE - 1),
                               "page.offset");
    llvm::Value *EndOff  = B.CreateAdd(PageOff,
                               llvm::ConstantInt::get(I64Ty, ByteRange),
                               "page.end");
    llvm::Value *Fits    = B.CreateICmpULT(EndOff,
                               llvm::ConstantInt::get(I64Ty, PAGE_SIZE),
                               "page.fits");

    // ── 2. Create a hoist block that runs mark_page once ──
    llvm::BasicBlock *HoistBB = llvm::BasicBlock::Create(Ctx, "mark.hoist", F,
                                                          L->getHeader());
    llvm::IRBuilder<> HB(HoistBB);
    llvm::Value *PageNo = HB.CreateLShr(PtrInt,
                              llvm::ConstantInt::get(I64Ty, PAGE_SHIFT));
    HB.CreateCall(MarkPageFn, {PageNo});
    HB.CreateBr(L->getHeader());

    // ── 3. Replace preheader's unconditional branch with a conditional one ──
    Preheader->getTerminator()->eraseFromParent();
    llvm::IRBuilder<> PB(Preheader);
    PB.CreateCondBr(Fits, HoistBB, L->getHeader());
    //               ↑ fits → hoist block (mark once)
    //                         ↑ doesn't fit → go straight to loop (mark per-iter)
}

void hoistBatchMarkPages(llvm::Value *Ptr, llvm::Loop *L,
                          const PageInvarianceResult &Result,
                          llvm::Function *MarkPageFn) {
    if (Result.tripCount == 0) return; // need a known TC for batch marking
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return;

    llvm::LLVMContext &Ctx = Preheader->getContext();
    auto *I64Ty = llvm::Type::getInt64Ty(Ctx);

    llvm::IRBuilder<> B(Preheader->getTerminator());

    // base_page = ptr >> PAGE_SHIFT
    llvm::Value *PtrInt  = B.CreatePtrToInt(Ptr, I64Ty);
    llvm::Value *BasePage = B.CreateLShr(PtrInt,
                                llvm::ConstantInt::get(I64Ty, PAGE_SHIFT),
                                "base.page");

    // Emit one mark_page call per page touched
    // (safe to unroll here since TC is known and typically small for GPU kernels)
    for (uint64_t i = 0; i < Result.tripCount; i++) {
        llvm::Value *PageNo = B.CreateAdd(BasePage,
                                  llvm::ConstantInt::get(I64Ty, i),
                                  "page.no." + std::to_string(i));
        B.CreateCall(MarkPageFn, {PageNo});
    }
}

llvm::Function *getMarkPageFn(llvm::Module &M) {
    auto *I64Ty = llvm::Type::getInt64Ty(M.getContext());
    auto *FnTy  = llvm::FunctionType::get(
                      llvm::Type::getVoidTy(M.getContext()),
                      {I64Ty}, // takes a page number
                      false);
    // getOrInsertFunction won't duplicate if it already exists
    return llvm::cast<llvm::Function>(
               M.getOrInsertFunction("MarkAccess", FnTy).getCallee());
}

llvm::Function *getBatchMarkAccessFn(llvm::Module &M) {
    auto &Ctx    = M.getContext();
    auto *I64Ty  = llvm::Type::getInt64Ty(Ctx);
    auto *VoidTy = llvm::Type::getVoidTy(Ctx);
    // BatchMarkAccess(base_addr: i64, stride: i64, count: i64) → void
    auto *FnTy = llvm::FunctionType::get(VoidTy, {I64Ty, I64Ty, I64Ty}, false);
    return llvm::cast<llvm::Function>(
        M.getOrInsertFunction("BatchMarkAccess", FnTy).getCallee());
}

// Emit a single BatchMarkAccess(base, stride, count) call in the loop preheader.
// Handles both compile-time-constant and runtime (loop-invariant) stride/TC by
// expanding all three arguments via SCEVExpander.
// When SCEV cannot compute the trip count (typical for GPU grid-stride loops whose
// bound is a kernel arg and start depends on blockIdx/threadIdx), falls back to
// emitting ceildiv(bound - start, step) as IR at the preheader.
static bool emitBatchMarkAccess(
        llvm::Value *Ptr, llvm::Loop *L,
        const PageInvarianceResult &Result,
        llvm::ScalarEvolution &SE,
        llvm::Function *BatchMarkFn,
        const llvm::DataLayout &DL) {
    if (!Result.stride) return false;

    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return false;

    auto &Ctx   = Preheader->getContext();
    auto *I64Ty = llvm::Type::getInt64Ty(Ctx);
    auto *PtrTy = llvm::PointerType::getUnqual(Ctx);

    // ── Base: start value of the pointer recurrence ───────────────────────────
    const llvm::SCEV *PtrSCEV  = SE.getSCEV(Ptr);
    const llvm::SCEV *BaseSCEV = PtrSCEV;
    if (auto *AR = llvm::dyn_cast<llvm::SCEVAddRecExpr>(PtrSCEV))
        BaseSCEV = AR->getStart();

    // ── Stride: sign-extend to i64 ────────────────────────────────────────────
    const llvm::SCEV *StrideSCEV =
        SE.getTruncateOrSignExtend(Result.stride, I64Ty);

    // ── Trip count: static → SCEV BTC → runtime recovery ────────────────────
    bool needRuntimeCount  = false;
    bool needInclusiveBound = false;
    const llvm::SCEV *IVStartSCEV = nullptr;
    const llvm::SCEV *IVStepSCEV  = nullptr;
    llvm::Value      *BoundValIR  = nullptr;

    const llvm::SCEV *CountSCEV = nullptr;
    if (Result.tripCount > 0) {
        CountSCEV = SE.getConstant(I64Ty, Result.tripCount);
    } else {
        const llvm::SCEV *BTC = SE.getBackedgeTakenCount(L);
        if (!llvm::isa<llvm::SCEVCouldNotCompute>(BTC)) {
            CountSCEV = SE.getAddExpr(
                SE.getTruncateOrZeroExtend(BTC, I64Ty),
                SE.getConstant(I64Ty, 1));
        } else {
            // SCEV can't model the trip count. Recover by inspecting the loop
            // exit branch: extract the IV AddRec and loop-invariant bound, then
            // emit ceildiv(bound - start, step) as IR at the preheader.
            llvm::BasicBlock *ExitingBB = L->getExitingBlock();
            if (!ExitingBB) {
                llvm::errs() << "[UVM] BatchMarkAccess: no single exiting block\n";
                return false;
            }
            auto *BI = llvm::dyn_cast<llvm::BranchInst>(ExitingBB->getTerminator());
            if (!BI || !BI->isConditional()) return false;
            auto *Cmp = llvm::dyn_cast<llvm::ICmpInst>(BI->getCondition());
            if (!Cmp) return false;

            // Loop continues on the true successor; invert predicate if needed.
            bool TrueIsLoop = L->contains(BI->getSuccessor(0));
            llvm::ICmpInst::Predicate Pred = Cmp->getPredicate();
            llvm::Value *LHS = Cmp->getOperand(0);
            llvm::Value *RHS = Cmp->getOperand(1);
            if (!TrueIsLoop)
                Pred = llvm::CmpInst::getInversePredicate(Pred);

            // Normalize to LHS < RHS (ULT or SLT).
            if (Pred == llvm::ICmpInst::ICMP_UGT || Pred == llvm::ICmpInst::ICMP_SGT ||
                Pred == llvm::ICmpInst::ICMP_UGE || Pred == llvm::ICmpInst::ICMP_SGE) {
                std::swap(LHS, RHS);
                Pred = llvm::CmpInst::getSwappedPredicate(Pred);
            }
            // Convert inclusive upper bound (SLE/ULE: a <= bound) to exclusive
            // (a < bound+1) so the ceildiv formula works uniformly.
            bool inclusiveBound = false;
            if (Pred == llvm::ICmpInst::ICMP_SLE || Pred == llvm::ICmpInst::ICMP_ULE) {
                inclusiveBound = true;
                Pred = (Pred == llvm::ICmpInst::ICMP_SLE)
                       ? llvm::ICmpInst::ICMP_SLT : llvm::ICmpInst::ICMP_ULT;
            }
            if (Pred != llvm::ICmpInst::ICMP_ULT && Pred != llvm::ICmpInst::ICMP_SLT &&
                Pred != llvm::ICmpInst::ICMP_NE)
                return false;

            // Bound (RHS = 'n') must be loop-invariant.
            if (!L->isLoopInvariant(RHS)) return false;
            BoundValIR = RHS;

            // LHS must be an affine AddRec over L so we can read start and step.
            auto *IvAR = llvm::dyn_cast<llvm::SCEVAddRecExpr>(SE.getSCEV(LHS));
            if (!IvAR || !IvAR->isAffine() || IvAR->getLoop() != L) return false;

            IVStartSCEV = SE.getTruncateOrZeroExtend(IvAR->getStart(), I64Ty);
            IVStepSCEV  = SE.getTruncateOrSignExtend(IvAR->getStepRecurrence(SE), I64Ty);
            needRuntimeCount  = true;
            needInclusiveBound = inclusiveBound;
            // Use a constant dummy for the hoisting invariance check; the real
            // runtime count (derived from kernel args) is invariant at every loop level.
            CountSCEV = SE.getConstant(I64Ty, 0);
        }
    }

    // ── Hoist: walk up to the outermost loop where all three are invariant ────
    llvm::Loop *HoistTarget = L;
    for (llvm::Loop *Parent = L->getParentLoop();
         Parent && Parent->getLoopPreheader() &&
         SE.isLoopInvariant(BaseSCEV,   Parent) &&
         SE.isLoopInvariant(StrideSCEV, Parent) &&
         SE.isLoopInvariant(CountSCEV,  Parent);
         Parent = Parent->getParentLoop()) {
        HoistTarget = Parent;
    }

    llvm::BasicBlock *HoistPreheader = HoistTarget->getLoopPreheader();
    if (!HoistPreheader) return false;

    if (HoistTarget != L)
        llvm::errs() << "[UVM] BatchMarkAccess hoisted depth "
                     << L->getLoopDepth() << " → "
                     << HoistTarget->getLoopDepth() << "\n";

    // ── Expand SCEVs and emit runtime count (if needed) ───────────────────────
    llvm::SCEVExpander Expander(SE, DL, "batch.mark");
    auto InsertIt = HoistPreheader->getTerminator()->getIterator();

    llvm::Value *BasePtr   = Expander.expandCodeFor(BaseSCEV,   PtrTy,  InsertIt);
    llvm::Value *StrideVal = Expander.expandCodeFor(StrideSCEV, I64Ty,  InsertIt);
    llvm::Value *CountVal;

    if (needRuntimeCount) {
        llvm::IRBuilder<> RB(HoistPreheader->getTerminator());
        llvm::Value *StartV = Expander.expandCodeFor(IVStartSCEV, I64Ty, InsertIt);
        llvm::Value *StepV  = Expander.expandCodeFor(IVStepSCEV,  I64Ty, InsertIt);
        llvm::Value *BoundV = RB.CreateZExtOrTrunc(BoundValIR, I64Ty, "bound.i64");
        // For inclusive upper bound (a <= bound), treat as a < bound+1.
        if (needInclusiveBound)
            BoundV = RB.CreateAdd(BoundV, llvm::ConstantInt::get(I64Ty, 1), "bound.inc");
        // ceildiv(Bound - Start, Step) = (Bound - Start + Step - 1) / Step
        llvm::Value *Range  = RB.CreateSub(BoundV, StartV, "rt.range");
        llvm::Value *Step1  = RB.CreateSub(StepV, llvm::ConstantInt::get(I64Ty, 1), "step.m1");
        llvm::Value *Ceil   = RB.CreateAdd(Range, Step1, "rt.ceil");
        llvm::Value *Div    = RB.CreateUDiv(Ceil, StepV, "rt.count.raw");
        // Clamp to 0 if start >= bound (thread starts past the array end).
        llvm::Value *Active = RB.CreateICmpULT(StartV, BoundV, "rt.active");
        CountVal = RB.CreateSelect(Active, Div,
                                   llvm::ConstantInt::get(I64Ty, 0), "rt.count");
        llvm::errs() << "[UVM] BatchMarkAccess: emitting runtime count for " << *PtrSCEV << "\n";
    } else {
        CountVal = Expander.expandCodeFor(CountSCEV, I64Ty, InsertIt);
    }

    llvm::IRBuilder<> B(HoistPreheader->getTerminator());
    llvm::Value *BaseInt = B.CreatePtrToInt(BasePtr, I64Ty, "batch.base");
    B.CreateCall(BatchMarkFn, {BaseInt, StrideVal, CountVal});
    return true;
}

llvm::Value *emitPageInvarianceRuntimeCheck(
        llvm::Value    *BasePtr,   // the loop-invariant base address
        uint64_t        byteRange, // stride * (TC - 1)
        llvm::IRBuilder<> &B,
        llvm::LLVMContext &Ctx) {

    auto *I64Ty    = llvm::Type::getInt64Ty(Ctx);
    auto *PAGE_MASK = llvm::ConstantInt::get(I64Ty, PAGE_SIZE - 1);
    auto *Range     = llvm::ConstantInt::get(I64Ty, byteRange);
    auto *PageSz    = llvm::ConstantInt::get(I64Ty, PAGE_SIZE);

    // offset_in_page = base & (PAGE_SIZE - 1)
    llvm::Value *BaseInt = B.CreatePtrToInt(BasePtr, I64Ty, "base.int");
    llvm::Value *Offset  = B.CreateAnd(BaseInt, PAGE_MASK, "page.offset");

    // fits = (offset + byteRange) < PAGE_SIZE
    llvm::Value *EndOffset = B.CreateAdd(Offset, Range, "page.end.offset");
    llvm::Value *Fits      = B.CreateICmpULT(EndOffset, PageSz, "page.fits");

    return Fits; // use this to guard hoisted instrumentation
}

    void processMemoryOp(llvm::Instruction &I, llvm::Loop *L,
                        llvm::ScalarEvolution &SE,
                        llvm::LoopInfo &LI) {
        llvm::Value *Ptr = nullptr;
        if (auto *Load  = llvm::dyn_cast<llvm::LoadInst>(&I))  Ptr = Load->getPointerOperand();
        if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&I)) Ptr = Store->getPointerOperand();
        if (!Ptr) return;

        // Skip non-UVM (addrspace 0 = generic/UVM on CUDA, 1 = global)
        // Use trueAddressSpace to see through addrspacecast (e.g. __shared__ AS3 → AS0).
        unsigned AS = trueAddressSpace(Ptr);
        if (AS != 0 && AS != 1) return;

        auto Result = checkPageInvariance(Ptr, L, SE);

        switch (Result.kind) {
        case PageInvariance::Invariant:
            // Safe to hoist mark_page() to preheader unconditionally
            llvm::errs() << "[UVM] Page-invariant (address invariant): " << *Ptr << "\n";
            hoistMarkPage(Ptr, L, MarkFuncPtr);
            break;

        case PageInvariance::MaybeInvariant:
            // Hoist behind runtime check
            llvm::errs() << "[UVM] Maybe page-invariant: range=" 
                        << Result.byteRange << " bytes\n";
            hoistMarkPageConditional(Ptr, L, Result.byteRange, MarkFuncPtr);
            break;

        case PageInvariance::Variant:
            // Keep instrumentation in-loop; optionally prefetch or
            // analyze the stride pattern for your 2-level table traversal
            llvm::errs() << "[UVM] Page-variant: stride=" 
                        << *Result.stride << " TC=" << Result.tripCount << "\n";
            // Bonus: if stride == PAGE_SIZE, you can compute accessed pages
            // statically and batch-mark them in the preheader
            if (isPageAlignedStride(Result.stride))
                hoistBatchMarkPages(Ptr, L, Result, MarkFuncPtr);
            break;

        case PageInvariance::Unknown:
            // Leave as-is — your existing per-op instrumentation handles it
            break;
        }
    }
                              
using namespace llvm;

class UvmTrackingPass : public PassInfoMixin<UvmTrackingPass> {
public:

PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    if (M.getTargetTriple().find("nvptx") == std::string::npos)
        return PreservedAnalyses::all();
    if (M.getSourceFileName().find("libMarkAccess") != std::string::npos)
        return PreservedAnalyses::all();

    fprintf(stderr, "[UvmPass] Running on module: %s\n",
            M.getSourceFileName().c_str());

    auto &Ctx = M.getContext();
    FunctionCallee MarkFunc = M.getOrInsertFunction(
        "MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    auto *BatchMarkFn = getBatchMarkAccessFn(M);

    // ── Get the FAM proxy so we can request function-level analyses ──
    auto &FAMProxy = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M);
    FunctionAnalysisManager &FAM = FAMProxy.getManager();

    bool Changed = false;

    for (Function &F : M) {
        if (!F.hasFnAttribute("target-features")) continue;
        if (!F.getFnAttribute("target-features").getValueAsString().contains("ptx")) continue;
        if (F.isDeclaration() || F.getName().contains("MarkAccess")) continue;

        fprintf(stderr, "[UvmPass] Processing function: %s\n",
                F.getName().str().c_str());

        // ── Now you can use function-level analyses ──
        auto &LI = FAM.getResult<LoopAnalysis>(F);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

        // ── PHASE ONE: Collect targets ──
        std::vector<Instruction*> Targets;
        for (auto &BB : F) {
            for (auto &Inst : BB) {
                Value *Ptr = nullptr;
                if (auto *LI = dyn_cast<LoadInst>(&Inst))  Ptr = LI->getPointerOperand();
                if (auto *SI = dyn_cast<StoreInst>(&Inst)) Ptr = SI->getPointerOperand();

                if (Ptr && Ptr->getType()->isPointerTy()) {
                    unsigned AS = trueAddressSpace(Ptr);
                    if (AS == 0 || AS == 1)
                        Targets.push_back(&Inst);
                }
            }
        }

        // Per-thread page cache: alloca in the entry block so each GPU thread
        // gets its own copy in registers instead of sharing a device global.
        // Created after Phase 1 so its own store won't end up in Targets.
        IRBuilder<> AllocaBuilder(&F.getEntryBlock(),
                                  F.getEntryBlock().getFirstInsertionPt());
        AllocaInst *LocalCacheVar = AllocaBuilder.CreateAlloca(
            Type::getInt64Ty(Ctx), nullptr, "last_page_local");
        AllocaBuilder.CreateStore(
            ConstantInt::get(Type::getInt64Ty(Ctx), ~0ULL), LocalCacheVar);

        // ── PHASE TWO: Instrument ──
        errs() << "[UvmPass] Found " << Targets.size() << " targets in "
               << F.getName() << "\n";
        for (Instruction *Inst : Targets) {
            Value *Ptr = isa<LoadInst>(Inst)
                             ? cast<LoadInst>(Inst)->getPointerOperand()
                             : cast<StoreInst>(Inst)->getPointerOperand();

            if (const DebugLoc &DL = Inst->getDebugLoc()) {
                errs() << "[UvmPass] Instrumenting " 
                       << DL->getFilename() << ":" << DL.getLine()
                       << " (" << (isa<LoadInst>(Inst) ? "load" : "store") << ")\n";
            } 

            // Find which loop this instruction is in (if any)
            Loop *L = LI.getLoopFor(Inst->getParent()); // ← new
            if (L) {
                errs() << "   [-] Analyzing loop at depth " << L->getLoopDepth()
                       << " for instruction: " << *Inst << "\n";
                // Try page invariance — hoist if possible
                auto Result = checkPageInvariance(Ptr, L, SE);
                


                if (Result.kind == PageInvariance::Invariant) {
                    auto SCEVStride = Result.stride ? Result.stride : nullptr;
                    if(Result.stride) {
                        errs() << "   [-] SCEV Stride: " << SCEVStride << "\n";
                    }
                    errs() << "[UVM] Page-invariant: " << *Ptr << "\n";

                    hoistMarkPage(Ptr, L, cast<Function>(MarkFunc.getCallee()));
                    continue; // skip per-instruction instrumentation below
                }
                if (Result.kind == PageInvariance::MaybeInvariant) {
                    errs() << "[UVM] Maybe page-invariant: range=" 
                           << Result.byteRange << " bytes\n";
                    hoistMarkPageConditional(Ptr, L, Result.byteRange,
                                            cast<Function>(MarkFunc.getCallee()));
                    continue;
                }
                if (Result.kind == PageInvariance::Variant && Result.stride) {
                    errs() << "[UVM] BatchMarkAccess: stride=" << *Result.stride
                           << " TC=" << Result.tripCount << "\n";
                    if (emitBatchMarkAccess(Ptr, L, Result, SE, BatchMarkFn,
                                           M.getDataLayout()))
                        continue;
                    else
                        errs() << "[UVM] BatchMarkAccess failed to emit for " << *Ptr << "\n";
                    // batch failed — fall through to per-instruction fallback
                }
                // Unknown → per-instruction fallback
            } 


                    // fallback: per-instruction instrumentation
            IRBuilder<> Builder(Inst);
            Value *AddrInt  = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
            Value *CurPage  = Builder.CreateLShr(AddrInt, 12);
            Value *LastPage = Builder.CreateLoad(Builder.getInt64Ty(), LocalCacheVar);
            Value *IsNewPage = Builder.CreateICmpNE(CurPage, LastPage);

            Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
            Builder.SetInsertPoint(ThenTerm);
            Builder.CreateCall(MarkFunc, {AddrInt});
            Builder.CreateStore(CurPage, LocalCacheVar);

            Changed = true;
            errs() << "   [+] Instrumented: " << *Inst << "\n";
        }
    }
    errs() << "[UvmPass] Finished processing module: " << M.getSourceFileName() << "\n";
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "UvmTrackingPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase) {
                    MPM.addPass(UvmTrackingPass());
                });
        }
    };
}