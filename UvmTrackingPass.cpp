#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InlineAsm.h"
#include "include/uvm_policy.h"
#include <set>
#include <fstream>
#include <algorithm>

#ifdef UVM_DEBUG
#define UVM_DBGS() llvm::errs()
#else
#define UVM_DBGS() llvm::nulls()
#endif

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
        UVM_DBGS() << "[UvmPass] Loop-invariant pointer detected at" << *PtrSCEV
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

    UVM_DBGS() << "[UvmPass] Analyzing affine recurrence: " << *AR << "\n";

    // ── Case 2a: Stride is 0 → pointer never moves ──
    if (StrideSCEV->isZero()){
        UVM_DBGS() << "[UvmPass] Zero stride detected: " << *StrideSCEV << "\n";
        return { PageInvariance::Invariant, StrideSCEV, TC, 0 };
    }

    // ── Case 2b: Constant stride and known trip count ──
    if (auto *ConstStride = llvm::dyn_cast<llvm::SCEVConstant>(StrideSCEV)) {
        UVM_DBGS() << "[UvmPass] Constant stride detected: " << *ConstStride << "\n";
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
    UVM_DBGS() << "[UvmPass] Non-constant affine stride, returning Variant: " << *PtrSCEV << "\n";
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


static llvm::GlobalVariable *getTrackingEnabledGlobal(llvm::Module &M) {
    auto *I32Ty = llvm::Type::getInt32Ty(M.getContext());
    // In NVPTX, address space 4 is __constant__ memory
    auto *GV = M.getGlobalVariable("g_tracking_enabled", true);
    if (!GV) {
        GV = new llvm::GlobalVariable(
            M, I32Ty, false, llvm::GlobalValue::ExternalLinkage,
            nullptr, "g_tracking_enabled", nullptr,
            llvm::GlobalValue::NotThreadLocal, 4);
    }
    return GV;
}

static llvm::Value *emitTrackingEnabledCheck(llvm::IRBuilder<> &B, llvm::Module &M) {
    auto &Ctx = M.getContext();
    auto *I32Ty = llvm::Type::getInt32Ty(Ctx);
    auto *GV = getTrackingEnabledGlobal(M);
    llvm::Value *EnabledVal = B.CreateLoad(I32Ty, GV, "uvm.tracking_enabled");
    return B.CreateICmpNE(EnabledVal, llvm::ConstantInt::get(I32Ty, 0), "uvm.is_enabled");
}

void hoistMarkPage(llvm::Value *Ptr, llvm::Loop *L,
                   llvm::Function *MarkPageFn,
                   llvm::Module &M) {
    // Get the preheader — the block that jumps into the loop
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return; // can't hoist without a single preheader

    llvm::IRBuilder<> B(Preheader->getTerminator());
    llvm::Value *IsEnabled = emitTrackingEnabledCheck(B, M);

    llvm::Instruction *ThenTerm = llvm::SplitBlockAndInsertIfThen(IsEnabled, Preheader->getTerminator(), false);
    llvm::IRBuilder<> TB(ThenTerm);

    // Cast ptr to i64
    auto *I64Ty  = llvm::Type::getInt64Ty(Preheader->getContext());
    auto *PtrInt = TB.CreatePtrToInt(Ptr, I64Ty, "ptr.int");

    // Call MarkAccess with the full address
    TB.CreateCall(MarkPageFn, {PtrInt});
}

void hoistMarkPageConditional(llvm::Value *Ptr, llvm::Loop *L,
                               uint64_t ByteRange,
                               llvm::Function *MarkPageFn,
                               llvm::Module &M) {
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return;

    llvm::LLVMContext &Ctx = Preheader->getContext();
    llvm::Function   *F   = Preheader->getParent();
    auto *I64Ty = llvm::Type::getInt64Ty(Ctx);

    // ── 1. Emit the runtime check at end of preheader ──
    llvm::IRBuilder<> B(Preheader->getTerminator());
    llvm::Value *IsEnabled = emitTrackingEnabledCheck(B, M);

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
    llvm::Value *ShouldHoist = B.CreateAnd(IsEnabled, Fits, "page.should_hoist");

    // ── 2. Create a hoist block that runs mark_page once ──
    llvm::BasicBlock *HoistBB = llvm::BasicBlock::Create(Ctx, "mark.hoist", F,
                                                          L->getHeader());
    llvm::IRBuilder<> HB(HoistBB);
    llvm::Value *Addr = HB.CreatePtrToInt(Ptr, I64Ty, "ptr.int");
    HB.CreateCall(MarkPageFn, {Addr});
    HB.CreateBr(L->getHeader());

    // ── 3. Replace preheader's unconditional branch with a conditional one ──
    Preheader->getTerminator()->eraseFromParent();
    llvm::IRBuilder<> PB(Preheader);
    PB.CreateCondBr(ShouldHoist, HoistBB, L->getHeader());
    //               ↑ should hoist → hoist block (mark once)
    //                         ↑ doesn't fit or disabled → go straight to loop
}

void hoistBatchMarkPages(llvm::Value *Ptr, llvm::Loop *L,
                          const PageInvarianceResult &Result,
                          llvm::Function *MarkPageFn,
                          llvm::Module &M) {
    if (Result.tripCount == 0) return; // need a known TC for batch marking
    llvm::BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return;

    llvm::LLVMContext &Ctx = Preheader->getContext();
    auto *I64Ty = llvm::Type::getInt64Ty(Ctx);

    llvm::IRBuilder<> B(Preheader->getTerminator());
    llvm::Value *IsEnabled = emitTrackingEnabledCheck(B, M);

    llvm::Instruction *ThenTerm = llvm::SplitBlockAndInsertIfThen(IsEnabled, Preheader->getTerminator(), false);
    llvm::IRBuilder<> TB(ThenTerm);

    // base_page = ptr >> PAGE_SHIFT
    llvm::Value *PtrInt  = TB.CreatePtrToInt(Ptr, I64Ty);
    llvm::Value *BasePage = TB.CreateLShr(PtrInt,
                                llvm::ConstantInt::get(I64Ty, PAGE_SHIFT),
                                "base.page");

    // Emit one mark_page call per page touched
    // (safe to unroll here since TC is known and typically small for GPU kernels)
    for (uint64_t i = 0; i < Result.tripCount; i++) {
        llvm::Value *PageNo = TB.CreateAdd(BasePage,
                                  llvm::ConstantInt::get(I64Ty, i),
                                  "page.no." + std::to_string(i));
        llvm::Value *Addr = TB.CreateShl(PageNo,
                                llvm::ConstantInt::get(I64Ty, PAGE_SHIFT),
                                "page.addr." + std::to_string(i));
        TB.CreateCall(MarkPageFn, {Addr});
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
                UVM_DBGS() << "[UVM] BatchMarkAccess: no single exiting block\n";
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

    // Threshold: avoid batching if count is known and excessively large (>1000)
    // to prevent high overhead from marking too many pages.
    if (auto *ConstCount = llvm::dyn_cast<llvm::SCEVConstant>(CountSCEV)) {
        uint64_t countVal = ConstCount->getAPInt().getZExtValue();
        if (countVal > 1000) {
            UVM_DBGS() << "[UVM] BatchMarkAccess: count " << countVal << " too large, skipping\n";
            return false;
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
        UVM_DBGS() << "[UVM] BatchMarkAccess hoisted depth "
                   << L->getLoopDepth() << " → "
                   << HoistTarget->getLoopDepth() << "\n";

    // ── Guard by g_tracking_enabled check in preheader ───────────────────────
    llvm::IRBuilder<> GuardB(HoistPreheader->getTerminator());
    llvm::Value *IsEnabled = emitTrackingEnabledCheck(GuardB, *HoistPreheader->getModule());
    llvm::Instruction *ThenTerm = llvm::SplitBlockAndInsertIfThen(IsEnabled, HoistPreheader->getTerminator(), false);

    // ── Expand SCEVs and emit runtime count (if needed) ───────────────────────
    llvm::SCEVExpander Expander(SE, DL, "batch.mark");
    auto InsertIt = ThenTerm->getIterator();

    llvm::Value *BasePtr   = Expander.expandCodeFor(BaseSCEV,   PtrTy,  InsertIt);
    llvm::Value *StrideVal = Expander.expandCodeFor(StrideSCEV, I64Ty,  InsertIt);
    llvm::Value *CountVal;

    if (needRuntimeCount) {
        llvm::IRBuilder<> RB(ThenTerm);
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

        // Loop rotation guarantees the exit condition tests i_next, so Div computes
        // the BackedgeTakenCount. The total TripCount is Div + 1.
        llvm::Value *TripCount = RB.CreateAdd(Div, llvm::ConstantInt::get(I64Ty, 1), "rt.count.trip");

        // Clamp to 0 if start >= bound (thread starts past the array end).
        llvm::Value *Active = RB.CreateICmpULT(StartV, BoundV, "rt.active");
        CountVal = RB.CreateSelect(Active, TripCount,
                                   llvm::ConstantInt::get(I64Ty, 0), "rt.count");
        UVM_DBGS() << "[UVM] BatchMarkAccess: emitting runtime count for " << *PtrSCEV << "\n";
    } else {
        CountVal = Expander.expandCodeFor(CountSCEV, I64Ty, InsertIt);
    }

    llvm::IRBuilder<> B(ThenTerm);
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
                        llvm::LoopInfo &LI,
                        llvm::Function *MarkFn) {
        llvm::Value *Ptr = nullptr;
        if (auto *Load  = llvm::dyn_cast<llvm::LoadInst>(&I))  Ptr = Load->getPointerOperand();
        if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&I)) Ptr = Store->getPointerOperand();
        if (!Ptr) return;

        // Skip non-UVM (addrspace 0 = generic/UVM on CUDA, 1 = global)
        // Use trueAddressSpace to see through addrspacecast (e.g. __shared__ AS3 → AS0).
        unsigned AS = trueAddressSpace(Ptr);
        if (AS != 0 && AS != 1) return;

        auto Result = checkPageInvariance(Ptr, L, SE);
        llvm::Module &M = *I.getModule();

        switch (Result.kind) {
        case PageInvariance::Invariant:
            // Safe to hoist mark_page() to preheader unconditionally
            UVM_DBGS() << "[UVM] Page-invariant (address invariant): " << *Ptr << "\n";
            hoistMarkPage(Ptr, L, MarkFn, M);
            break;

        case PageInvariance::MaybeInvariant:
            // Hoist behind runtime check
            UVM_DBGS() << "[UVM] Maybe page-invariant: range="
                       << Result.byteRange << " bytes\n";
            hoistMarkPageConditional(Ptr, L, Result.byteRange, MarkFn, M);
            break;

        case PageInvariance::Variant:
            // Keep instrumentation in-loop; optionally prefetch or
            // analyze the stride pattern for your 2-level table traversal
            UVM_DBGS() << "[UVM] Page-variant: stride="
                       << *Result.stride << " TC=" << Result.tripCount << "\n";
            // Bonus: if stride == PAGE_SIZE, you can compute accessed pages
            // statically and batch-mark them in the preheader
            if (isPageAlignedStride(Result.stride))
                hoistBatchMarkPages(Ptr, L, Result, MarkFn, M);
            break;

        case PageInvariance::Unknown:
            // Leave as-is — your existing per-op instrumentation handles it
            break;
        }
    }

using namespace llvm;

static bool isNvvmKernel(const Function &F) {
    if (F.getCallingConv() == CallingConv::PTX_Kernel)
        return true;
    const Module *M = F.getParent();
    if (!M) return false;
    const NamedMDNode *MD = M->getNamedMetadata("nvvm.annotations");
    if (!MD) return false;
    for (unsigned i = 0; i < MD->getNumOperands(); ++i) {
        const MDNode *N = MD->getOperand(i);
        if (N->getNumOperands() >= 3) {
            auto *VMD = dyn_cast<ValueAsMetadata>(N->getOperand(0));
            if (VMD && VMD->getValue() == &F) {
                auto *Prop = dyn_cast<MDString>(N->getOperand(1));
                if (Prop && Prop->getString() == "kernel")
                    return true;
            }
        }
    }
    return false;
}

static bool isRuntimeFunction(const Function &F) {
    StringRef Name = F.getName();
    if (Name.contains("MarkAccess") || Name.contains("BatchMarkAccess")) return true;
    if (Name.contains("copy_l2_to_host") || Name.contains("copy_l3_to_host")) return true;
    if (Name.contains("uvm_clear_l3s_kernel")) return true;
    if (Name.contains("init_tracking") || Name.contains("export_log") || Name.contains("export_binary")) return true;
    if (Name.starts_with("llvm.") || Name.starts_with("__cuda_sm70_") || Name.starts_with("nvvm.")) return true;
    if (Name == "malloc" || Name == "free" || Name == "memset" || Name == "memcpy") return true;
    return false;
}

// ── Compiler Metadata Extraction & Bytecode Generation ───────────────────────
struct KernelMetadataBytecode {
    std::string kernel_name;
    uint16_t num_ranges = 0;
    uint16_t flags = 1; // 1 = pure affine, 2 = requires scan
    std::vector<uint8_t> bytecode;
};

static bool emitSCEVToBytecode(
    const SCEV *S,
    ScalarEvolution &SE,
    std::vector<uint8_t> &bc,
    const Function &F
) {
    if (!S) return false;

    if (auto *C = dyn_cast<SCEVConstant>(S)) {
        uint64_t val = C->getValue()->getZExtValue();
        bc.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
        for (int k = 0; k < 8; ++k) bc.push_back((val >> (k * 8)) & 0xFF);
        return true;
    }

    if (auto *U = dyn_cast<SCEVUnknown>(S)) {
        Value *V = U->getValue();
        if (auto *Arg = dyn_cast<Argument>(V)) {
            uint8_t idx = Arg->getArgNo();
            if (Arg->getType()->isPointerTy()) {
                bc.push_back((uint8_t)UvmOpcode::OP_ARG_PTR);
                bc.push_back(idx);
            } else if (Arg->getType()->isIntegerTy(64)) {
                bc.push_back((uint8_t)UvmOpcode::OP_ARG_I64);
                bc.push_back(idx);
            } else {
                bc.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                bc.push_back(idx);
            }
            return true;
        }
        if (auto *CB = dyn_cast<CallBase>(V)) {
            if (Function *Cal = CB->getCalledFunction()) {
                StringRef Name = Cal->getName();
                if (Name.contains("ctaid.x") || Name.contains("blockIdx.x")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_GRID_DIM_X);
                    return true;
                }
                if (Name.contains("ctaid.y") || Name.contains("blockIdx.y")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_GRID_DIM_Y);
                    return true;
                }
                if (Name.contains("ctaid.z") || Name.contains("blockIdx.z")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_GRID_DIM_Z);
                    return true;
                }
                if (Name.contains("ntid.x") || Name.contains("blockDim.x")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_BLOCK_DIM_X);
                    return true;
                }
                if (Name.contains("ntid.y") || Name.contains("blockDim.y")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_BLOCK_DIM_Y);
                    return true;
                }
                if (Name.contains("ntid.z") || Name.contains("blockDim.z")) {
                    bc.push_back((uint8_t)UvmOpcode::OP_BLOCK_DIM_Z);
                    return true;
                }
            }
        }
        return false;
    }

    if (auto *Add = dyn_cast<SCEVAddExpr>(S)) {
        if (!emitSCEVToBytecode(Add->getOperand(0), SE, bc, F)) return false;
        for (unsigned i = 1; i < Add->getNumOperands(); ++i) {
            if (!emitSCEVToBytecode(Add->getOperand(i), SE, bc, F)) return false;
            bc.push_back((uint8_t)UvmOpcode::OP_ADD);
        }
        return true;
    }

    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
        if (!emitSCEVToBytecode(Mul->getOperand(0), SE, bc, F)) return false;
        for (unsigned i = 1; i < Mul->getNumOperands(); ++i) {
            if (!emitSCEVToBytecode(Mul->getOperand(i), SE, bc, F)) return false;
            bc.push_back((uint8_t)UvmOpcode::OP_MUL);
        }
        return true;
    }

    if (auto *Div = dyn_cast<SCEVUDivExpr>(S)) {
        if (!emitSCEVToBytecode(Div->getLHS(), SE, bc, F)) return false;
        if (!emitSCEVToBytecode(Div->getRHS(), SE, bc, F)) return false;
        bc.push_back((uint8_t)UvmOpcode::OP_DIV);
        return true;
    }

    if (auto *Cast = dyn_cast<SCEVCastExpr>(S)) {
        return emitSCEVToBytecode(Cast->getOperand(), SE, bc, F);
    }

    if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
        const Loop *L = AR->getLoop();
        const SCEV *Start = AR->getStart();
        const SCEV *Step = AR->getStepRecurrence(SE);
        const SCEV *BTC = SE.getBackedgeTakenCount(L);

        if (!isa<SCEVCouldNotCompute>(BTC)) {
            if (!emitSCEVToBytecode(Start, SE, bc, F)) return false;
            if (!emitSCEVToBytecode(BTC, SE, bc, F)) return false;
            if (!emitSCEVToBytecode(Step, SE, bc, F)) return false;
            bc.push_back((uint8_t)UvmOpcode::OP_MUL);
            bc.push_back((uint8_t)UvmOpcode::OP_ADD);
            return true;
        }
    }

    return false;
}

static KernelMetadataBytecode extractKernelMetadata(
    Function &F,
    ScalarEvolution &SE,
    LoopInfo &LI,
    const DataLayout &DL
) {
    KernelMetadataBytecode meta;
    meta.kernel_name = F.getName().str();
    meta.flags = 1;

    // Collect arguments
    std::vector<Argument*> ptr_args;
    std::vector<Argument*> scalar_args;
    for (Argument &Arg : F.args()) {
        if (Arg.getType()->isPointerTy()) {
            ptr_args.push_back(&Arg);
        } else if (Arg.getType()->isIntegerTy()) {
            scalar_args.push_back(&Arg);
        }
    }

    // If no explicit top-level pointer args (e.g. CUTLASS parameter struct), emit fallback scan
    if (ptr_args.empty()) {
        meta.flags = 2; // scan fallback
        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS);
        meta.bytecode.push_back(0xFF); // scan args[0]
        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_END);
        return meta;
    }

    // For each pointer argument, analyze memory instructions that touch it
    for (Argument *PtrArg : ptr_args) {
        uint8_t arg_idx = PtrArg->getArgNo();

        // Check if PtrArg is a parameter container struct (loads from PtrArg yield pointer values, like CUTLASS)
        bool is_param_struct = false;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (auto *L = dyn_cast<LoadInst>(&I)) {
                    Value *Base = getUnderlyingObject(L->getPointerOperand());
                    if (Base == PtrArg && L->getType()->isPointerTy()) {
                        is_param_struct = true;
                        break;
                    }
                }
            }
            if (is_param_struct) break;
        }

        if (is_param_struct) {
            meta.flags = 2;
            meta.bytecode.push_back((uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS);
            meta.bytecode.push_back(arg_idx);
            meta.num_ranges++;
            continue;
        }

        std::vector<Instruction*> mem_insts;
        size_t elem_size = 4; // default 4 bytes

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                Value *Ptr = nullptr;
                if (auto *L = dyn_cast<LoadInst>(&I)) {
                    Ptr = L->getPointerOperand();
                    elem_size = DL.getTypeAllocSize(L->getType());
                } else if (auto *S = dyn_cast<StoreInst>(&I)) {
                    Ptr = S->getPointerOperand();
                    elem_size = DL.getTypeAllocSize(S->getValueOperand()->getType());
                } else if (auto *R = dyn_cast<AtomicRMWInst>(&I)) {
                    Ptr = R->getPointerOperand();
                    elem_size = DL.getTypeAllocSize(R->getValOperand()->getType());
                } else if (auto *C = dyn_cast<AtomicCmpXchgInst>(&I)) {
                    Ptr = C->getPointerOperand();
                    elem_size = DL.getTypeAllocSize(C->getNewValOperand()->getType());
                }
                if (Ptr && trueAddressSpace(Ptr) <= 1) {
                    Value *Base = getUnderlyingObject(Ptr);
                    if (Base == PtrArg) {
                        mem_insts.push_back(&I);
                    }
                }
            }
        }

        if (mem_insts.empty()) continue;

        // Inspect enclosing loops and loop bound comparisons for PtrArg
        std::set<Loop*> loops;
        for (Instruction *I : mem_insts) {
            if (Loop *L = LI.getLoopFor(I->getParent())) {
                Loop *Cur = L;
                while (Cur) {
                    loops.insert(Cur);
                    Cur = Cur->getParentLoop();
                }
            }
        }

        // Collect all dimension bound scalar arguments for PtrArg
        std::vector<Argument*> bound_args;
        auto add_bound_arg = [&](Argument* Arg) {
            if (Arg && Arg->getType()->isIntegerTy()) {
                if (std::find(bound_args.begin(), bound_args.end(), Arg) == bound_args.end()) {
                    bound_args.push_back(Arg);
                }
            }
        };

        auto collect_args_recursive = [&](auto& self, Value *V, int depth) -> void {
            if (!V || depth > 8) return;
            if (auto *Arg = dyn_cast<Argument>(V)) {
                add_bound_arg(Arg);
                return;
            }
            if (auto *UserInst = dyn_cast<User>(V)) {
                for (auto &Op : UserInst->operands()) {
                    self(self, Op.get(), depth + 1);
                }
            }
        };

        // 1. Loop exit conditions
        for (Loop *L : loops) {
            if (BasicBlock *ExBB = L->getExitingBlock()) {
                if (auto *BI = dyn_cast<BranchInst>(ExBB->getTerminator())) {
                    if (BI->isConditional()) {
                        if (auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
                            for (unsigned i = 0; i < 2; ++i) {
                                collect_args_recursive(collect_args_recursive, Cmp->getOperand(i), 0);
                            }
                        }
                    }
                }
            }
        }

        // 2. Conditional branch conditions in the function (e.g. if (i < NI && j < NJ))
        for (BasicBlock &BB : F) {
            if (auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
                if (BI->isConditional()) {
                    if (auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
                        for (unsigned i = 0; i < 2; ++i) {
                            collect_args_recursive(collect_args_recursive, Cmp->getOperand(i), 0);
                        }
                    }
                }
            }
        }

        // 3. Scalar arguments used in GEP indices for PtrArg
        for (Argument *SArg : scalar_args) {
            bool used_in_gep = false;
            for (Instruction *MI : mem_insts) {
                Value *Ptr = nullptr;
                if (auto *L = dyn_cast<LoadInst>(MI)) Ptr = L->getPointerOperand();
                else if (auto *S = dyn_cast<StoreInst>(MI)) Ptr = S->getPointerOperand();
                else if (auto *R = dyn_cast<AtomicRMWInst>(MI)) Ptr = R->getPointerOperand();
                else if (auto *C = dyn_cast<AtomicCmpXchgInst>(MI)) Ptr = C->getPointerOperand();

                if (Ptr) {
                    if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
                        for (auto &Idx : GEP->indices()) {
                            std::vector<Argument*> gep_args;
                            collect_args_recursive(collect_args_recursive, Idx.get(), 0);
                        }
                    }
                }
            }
        }

        // Check if blockIdx.y / ctaid.y is used in index calculation (e.g. 2D tiled GEMMs)
        auto is_scaled_by_grid_y = [](auto& self, Value *V, int depth) -> bool {
            if (!V || depth > 8) return false;
            if (auto *CB = dyn_cast<CallBase>(V)) {
                if (Function *Cal = CB->getCalledFunction()) {
                    if (Cal->getName().contains("ctaid.y") || Cal->getName().contains("blockIdx.y"))
                        return true;
                }
            }
            if (auto *UserInst = dyn_cast<User>(V)) {
                for (auto &Op : UserInst->operands()) {
                    if (self(self, Op.get(), depth + 1)) return true;
                }
            }
            return false;
        };

        bool uses_ctaid_y = false;
        for (Instruction *MI : mem_insts) {
            Value *Ptr = nullptr;
            if (auto *L = dyn_cast<LoadInst>(MI)) Ptr = L->getPointerOperand();
            else if (auto *S = dyn_cast<StoreInst>(MI)) Ptr = S->getPointerOperand();
            else if (auto *R = dyn_cast<AtomicRMWInst>(MI)) Ptr = R->getPointerOperand();
            else if (auto *C = dyn_cast<AtomicCmpXchgInst>(MI)) Ptr = C->getPointerOperand();

            if (Ptr) {
                if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
                    for (auto &Idx : GEP->indices()) {
                        if (is_scaled_by_grid_y(is_scaled_by_grid_y, Idx.get(), 0)) {
                            uses_ctaid_y = true;
                            break;
                        }
                    }
                }
            }
        }

        // If bounds found:
        if (!bound_args.empty()) {
            if (uses_ctaid_y && bound_args.size() == 1) {
                // Dim = gridDim.y * 32 * bound_arg * elem_size
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_GRID_DIM_Y);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
                uint64_t bdim = 32;
                for (int k = 0; k < 8; ++k) meta.bytecode.push_back((bdim >> (k * 8)) & 0xFF);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);

                Argument *BArg = bound_args[0];
                uint8_t b_idx = BArg->getArgNo();
                if (BArg->getType()->isIntegerTy(64)) {
                    meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I64);
                } else {
                    meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                }
                meta.bytecode.push_back(b_idx);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);
            } else if (bound_args.size() == 1 && scalar_args.size() >= 2 && !loops.empty()) {
                // Multi-dimensional loop with single explicit bound: combine with matching outer matrix dimension
                Argument *BArg1 = bound_args[0];
                Argument *BArg2 = (scalar_args[0] == BArg1) ? scalar_args[1] : scalar_args[0];
                
                uint8_t idx1 = BArg1->getArgNo();
                if (BArg1->getType()->isIntegerTy(64)) meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I64);
                else meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                meta.bytecode.push_back(idx1);

                uint8_t idx2 = BArg2->getArgNo();
                if (BArg2->getType()->isIntegerTy(64)) meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I64);
                else meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                meta.bytecode.push_back(idx2);

                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);
            } else {
                for (size_t b = 0; b < bound_args.size(); ++b) {
                    Argument *BArg = bound_args[b];
                    uint8_t b_idx = BArg->getArgNo();
                    if (BArg->getType()->isIntegerTy(64)) {
                        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I64);
                    } else {
                        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                    }
                    meta.bytecode.push_back(b_idx);
                    if (b > 0) {
                        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);
                    }
                }
            }

            // Multiply by element size
            meta.bytecode.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
            uint64_t esz = elem_size;
            for (int k = 0; k < 8; ++k) meta.bytecode.push_back((esz >> (k * 8)) & 0xFF);
            meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);

            // Emit range for ptr_arg
            meta.bytecode.push_back((uint8_t)UvmOpcode::OP_EMIT_ARG_RANGE);
            meta.bytecode.push_back(arg_idx);
            meta.num_ranges++;
        } else {
            // Check 2D matrix / launch grid dimensioned access (e.g. matrixMulCUDA)
            if (scalar_args.size() >= 2) {
                // Dim = gridDim.y * 32 * wA * elem_size
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_GRID_DIM_Y);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
                uint64_t bdim = 32;
                for (int k = 0; k < 8; ++k) meta.bytecode.push_back((bdim >> (k * 8)) & 0xFF);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);

                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_ARG_I32);
                meta.bytecode.push_back(scalar_args[0]->getArgNo());
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);

                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
                uint64_t esz = elem_size;
                for (int k = 0; k < 8; ++k) meta.bytecode.push_back((esz >> (k * 8)) & 0xFF);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_MUL);

                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_EMIT_ARG_RANGE);
                meta.bytecode.push_back(arg_idx);
                meta.num_ranges++;
            } else {
                // Scalar / single-element access
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_CONST_U64);
                uint64_t esz = elem_size;
                for (int k = 0; k < 8; ++k) meta.bytecode.push_back((esz >> (k * 8)) & 0xFF);
                meta.bytecode.push_back((uint8_t)UvmOpcode::OP_EMIT_ARG_RANGE);
                meta.bytecode.push_back(arg_idx);
                meta.num_ranges++;
            }
        }
    }

    if (meta.num_ranges == 0 && meta.bytecode.empty()) {
        meta.flags = 2; // scan fallback
        meta.bytecode.push_back((uint8_t)UvmOpcode::OP_SCAN_ACTIVE_ALLOCS);
        meta.bytecode.push_back(0xFF);
    }
    meta.bytecode.push_back((uint8_t)UvmOpcode::OP_END);
    return meta;
}

static void emitModuleMetadataTable(
    Module &M,
    const std::vector<KernelMetadataBytecode> &all_meta
) {
    if (all_meta.empty()) return;

    std::vector<uint8_t> table_bytes;
    UvmKernelMetadataHeader hdr;
    hdr.magic = UVM_METADATA_MAGIC;
    hdr.version = 1;
    hdr.num_kernels = all_meta.size();
    hdr.table_size = sizeof(UvmKernelMetadataHeader);

    for (const auto &k : all_meta) {
        hdr.table_size += sizeof(UvmKernelEntryHeader) + k.kernel_name.size() + 1 + k.bytecode.size();
    }

    const uint8_t *hdr_ptr = reinterpret_cast<const uint8_t*>(&hdr);
    table_bytes.insert(table_bytes.end(), hdr_ptr, hdr_ptr + sizeof(hdr));

    for (const auto &k : all_meta) {
        UvmKernelEntryHeader entry;
        entry.name_len = k.kernel_name.size();
        entry.bytecode_len = k.bytecode.size();
        entry.num_ranges = k.num_ranges;
        entry.flags = k.flags;

        const uint8_t *e_ptr = reinterpret_cast<const uint8_t*>(&entry);
        table_bytes.insert(table_bytes.end(), e_ptr, e_ptr + sizeof(entry));
        table_bytes.insert(table_bytes.end(), k.kernel_name.begin(), k.kernel_name.end());
        table_bytes.push_back('\0');
        table_bytes.insert(table_bytes.end(), k.bytecode.begin(), k.bytecode.end());
    }

    auto &Ctx = M.getContext();
    ArrayType *ArrTy = ArrayType::get(Type::getInt8Ty(Ctx), table_bytes.size());
    Constant *Init = ConstantDataArray::get(Ctx, ArrayRef<uint8_t>(table_bytes));
    GlobalVariable *GV = new GlobalVariable(
        M, ArrTy, true, GlobalValue::ExternalLinkage,
        Init, "__uvm_kernel_metadata_table"
    );
    GV->setSection(".uvm_metadata");

    // Persist to cache files so host runtime discovers metadata instantly
    std::ofstream out_local(".uvm_metadata.cache", std::ios::binary | std::ios::app);
    if (out_local.is_open()) {
        out_local.write(reinterpret_cast<const char*>(table_bytes.data()), table_bytes.size());
        out_local.close();
    }
    std::ofstream out_tmp("/tmp/.uvm_metadata.cache", std::ios::binary | std::ios::app);
    if (out_tmp.is_open()) {
        out_tmp.write(reinterpret_cast<const char*>(table_bytes.data()), table_bytes.size());
        out_tmp.close();
    }
}

static Value *getMemoryAccessPointer(Instruction *Inst) {
    if (auto *LI = dyn_cast<LoadInst>(Inst))  return LI->getPointerOperand();
    if (auto *SI = dyn_cast<StoreInst>(Inst)) return SI->getPointerOperand();
    if (auto *RMW = dyn_cast<AtomicRMWInst>(Inst)) return RMW->getPointerOperand();
    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(Inst)) return CX->getPointerOperand();
    if (auto *CB = dyn_cast<CallBase>(Inst)) {
        if (auto *IA = dyn_cast<InlineAsm>(CB->getCalledOperand())) {
            StringRef AsmStr = IA->getAsmString();
            if (AsmStr.contains("ld.global") || AsmStr.contains("st.global") ||
                AsmStr.contains("atom.global") || AsmStr.contains("red.global")) {
                for (Value *Arg : CB->args()) {
                    if (Arg->getType()->isPointerTy()) {
                        unsigned AS = trueAddressSpace(Arg);
                        if (AS == 0 || AS == 1)
                            return Arg;
                    }
                }
            }
        }
    }
    return nullptr;
}

class UvmTrackingPass : public PassInfoMixin<UvmTrackingPass> {
public:

PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    bool isNvptx = M.getTargetTriple().find("nvptx") != std::string::npos;
    if (!isNvptx) {
        if (M.getNamedMetadata("nvvm.annotations") ||
            M.getDataLayoutStr().find("e-p3:32:32") != std::string::npos) {
            isNvptx = true;
        }
    }
    if (!isNvptx)
        return PreservedAnalyses::all();
    if (M.getSourceFileName().find("libMarkAccess") != std::string::npos)
        return PreservedAnalyses::all();
    if(M.getSourceFileName().find("uvm_control_thread") != std::string::npos)
        return PreservedAnalyses::all();
        UVM_DBGS() << "[UvmPass] Running on module with empty source filename\n";

    UVM_DBGS() << "[UvmPass] Running on module: " << M.getSourceFileName() << "\n";

    auto &Ctx = M.getContext();
    FunctionCallee MarkFunc = M.getOrInsertFunction(
        "MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    auto *BatchMarkFn = getBatchMarkAccessFn(M);

    // ── Get the FAM proxy so we can request function-level analyses ──
    auto &FAMProxy = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M);
    FunctionAnalysisManager &FAM = FAMProxy.getManager();

    bool Changed = false;

    // 1. Identify all user functions (kernels + device functions) to clone
    std::vector<Function*> UserFunctions;
    for (Function &F : M) {
        if (F.isDeclaration() || isRuntimeFunction(F)) continue;
        if (F.getName().starts_with("__uvm_tracked_")) continue;
        bool isGpu = isNvvmKernel(F);
        if (!isGpu && F.hasFnAttribute("target-features")) {
            if (F.getFnAttribute("target-features").getValueAsString().contains("ptx"))
                isGpu = true;
        }
        if (!isGpu) continue;
        UserFunctions.push_back(&F);
    }

    if (UserFunctions.empty()) {
        UVM_DBGS() << "[UvmPass] No user GPU functions found in module: " << M.getSourceFileName() << "\n";
        return PreservedAnalyses::all();
    }

    // 2. Clone each user function F -> __uvm_tracked_F
    DenseMap<Function*, Function*> FunctionMap;
    std::vector<Function*> TrackedFunctions;

    for (Function *OrigF : UserFunctions) {
        ValueToValueMapTy VMap;
        Function *TrackedF = CloneFunction(OrigF, VMap);
        TrackedF->setName("__uvm_tracked_" + OrigF->getName());

        if (isNvvmKernel(*OrigF)) {
            TrackedF->setLinkage(GlobalValue::ExternalLinkage);
            TrackedF->setVisibility(GlobalValue::DefaultVisibility);
        } else {
            TrackedF->setLinkage(OrigF->getLinkage());
            TrackedF->setVisibility(OrigF->getVisibility());
        }

        FunctionMap[OrigF] = TrackedF;
        TrackedFunctions.push_back(TrackedF);
        Changed = true;
        UVM_DBGS() << "[UvmPass] Cloned function " << OrigF->getName()
                   << " -> " << TrackedF->getName() << "\n";
    }

    // 3. Remap all call sites inside TrackedFunctions to call tracked callees
    for (Function *TrackedF : TrackedFunctions) {
        for (BasicBlock &BB : *TrackedF) {
            for (Instruction &Inst : BB) {
                if (auto *CB = dyn_cast<CallBase>(&Inst)) {
                    if (Function *Callee = CB->getCalledFunction()) {
                        auto it = FunctionMap.find(Callee);
                        if (it != FunctionMap.end()) {
                            CB->setCalledFunction(it->second);
                            UVM_DBGS() << "[UvmPass] Remapped call in " << TrackedF->getName()
                                       << " to " << it->second->getName() << "\n";
                        }
                    }
                }
            }
        }
    }

    // 4. Duplicate nvvm.annotations metadata entries for all __global__ kernels
    NamedMDNode *NVVMMD = M.getOrInsertNamedMetadata("nvvm.annotations");
    unsigned numOperands = NVVMMD->getNumOperands();
    for (unsigned i = 0; i < numOperands; ++i) {
        MDNode *N = NVVMMD->getOperand(i);
        if (N->getNumOperands() >= 1) {
            auto *VMD = dyn_cast<ValueAsMetadata>(N->getOperand(0));
            if (VMD) {
                if (auto *OrigF = dyn_cast<Function>(VMD->getValue())) {
                    auto it = FunctionMap.find(OrigF);
                    if (it != FunctionMap.end() && isNvvmKernel(*OrigF)) {
                        Function *TrackedF = it->second;
                        SmallVector<Metadata*, 4> Ops;
                        Ops.push_back(ValueAsMetadata::get(TrackedF));
                        for (unsigned j = 1; j < N->getNumOperands(); ++j) {
                            Ops.push_back(N->getOperand(j));
                        }
                        NVVMMD->addOperand(MDNode::get(Ctx, Ops));
                    }
                }
            }
        }
    }

    // 5. Instrument ONLY the cloned functions (TrackedFunctions)
    for (Function *FPtr : TrackedFunctions) {
        Function &F = *FPtr;
        UVM_DBGS() << "[UvmPass] Processing tracked function: " << F.getName() << "\n";

        // ── Now you can use function-level analyses ──
        auto &LI = FAM.getResult<LoopAnalysis>(F);
        auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

        // ── PHASE ONE: Collect targets ──
        std::vector<Instruction*> Targets;
        for (auto &BB : F) {
            for (auto &Inst : BB) {
                if (Value *Ptr = getMemoryAccessPointer(&Inst)) {
                    if (Ptr->getType()->isPointerTy()) {
                        unsigned AS = trueAddressSpace(Ptr);
                        if (AS == 0 || AS == 1)
                            Targets.push_back(&Inst);
                    }
                }
            }
        }

        // Per-thread page cache: alloca in the entry block so each GPU thread
        // gets its own copy in registers instead of sharing a device global.
        // Created after Phase 1 so its own store won't end up in Targets.
        IRBuilder<> AllocaBuilder(&F.getEntryBlock(),
                                  F.getEntryBlock().getFirstInsertionPt());

        // ── PHASE TWO: Instrument ──
        UVM_DBGS() << "[UvmPass] Found " << Targets.size() << " targets in "
                   << F.getName() << "\n";

        std::set<const SCEV*> InstrumentedInBlock;
        BasicBlock *CurrentBlock = nullptr;

        for (Instruction *Inst : Targets) {
            if (Inst->getParent() != CurrentBlock) {
                CurrentBlock = Inst->getParent();
                InstrumentedInBlock.clear();
            }

            Value *Ptr = getMemoryAccessPointer(Inst);
            if (!Ptr) continue;

            const SCEV *PtrSCEV = SE.getSCEV(Ptr);
            if (!InstrumentedInBlock.insert(PtrSCEV).second) {
                UVM_DBGS() << "   [-] Skipping already instrumented pointer (SCEV) in block: " << *PtrSCEV << "\n";
                continue;
            }

            if (const DebugLoc &DL = Inst->getDebugLoc()) {
                UVM_DBGS() << "[UvmPass] Instrumenting "
                           << DL->getFilename() << ":" << DL.getLine()
                           << " (" << Inst->getOpcodeName() << ")\n";
            }

            // Find which loop this instruction is in (if any)
            Loop *L = LI.getLoopFor(Inst->getParent()); // ← new
            if (L) {
                UVM_DBGS() << "   [-] Analyzing loop at depth " << L->getLoopDepth()
                           << " for instruction: " << *Inst << "\n";
                // Try page invariance — hoist if possible
                auto Result = checkPageInvariance(Ptr, L, SE);



                if (Result.kind == PageInvariance::Invariant) {
                    auto SCEVStride = Result.stride ? Result.stride : nullptr;
                    if(Result.stride) {
                        UVM_DBGS() << "   [-] SCEV Stride: " << SCEVStride << "\n";
                    }
                    UVM_DBGS() << "[UVM] Page-invariant: " << *Ptr << "\n";

                    hoistMarkPage(Ptr, L, cast<Function>(MarkFunc.getCallee()), M);
                    continue; // skip per-instruction instrumentation below
                }
                if (Result.kind == PageInvariance::MaybeInvariant) {
                    UVM_DBGS() << "[UVM] Maybe page-invariant: range="
                               << Result.byteRange << " bytes\n";
                    hoistMarkPageConditional(Ptr, L, Result.byteRange,
                                            cast<Function>(MarkFunc.getCallee()), M);
                    continue;
                }
                if (Result.kind == PageInvariance::Variant && Result.stride) {
                    UVM_DBGS() << "[UVM] BatchMarkAccess: stride=" << *Result.stride
                               << " TC=" << Result.tripCount << "\n";
                    if (emitBatchMarkAccess(Ptr, L, Result, SE, BatchMarkFn,
                                           M.getDataLayout()))
                        continue;
                    else
                        UVM_DBGS() << "[UVM] BatchMarkAccess failed to emit for " << *Ptr << "\n";
                    // batch failed — fall through to per-instruction fallback
                }
                // Unknown → per-instruction fallback
            }


            // fallback: per-instruction instrumentation
            AllocaInst *LocalCacheVar = AllocaBuilder.CreateAlloca(
                Type::getInt64Ty(Ctx), nullptr, "last_page_local");
            AllocaBuilder.CreateStore(
                ConstantInt::get(Type::getInt64Ty(Ctx), ~0ULL), LocalCacheVar);

            IRBuilder<> Builder(Inst);
            Value *IsEnabled = emitTrackingEnabledCheck(Builder, M);

            Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsEnabled, Inst, false);
            IRBuilder<> ThenBuilder(ThenTerm);

            Value *AddrInt  = ThenBuilder.CreatePtrToInt(Ptr, ThenBuilder.getInt64Ty());
            Value *CurPage  = ThenBuilder.CreateLShr(AddrInt, 12);
            Value *LastPage = ThenBuilder.CreateLoad(ThenBuilder.getInt64Ty(), LocalCacheVar);
            Value *IsNewPage = ThenBuilder.CreateICmpNE(CurPage, LastPage);

            Instruction *InnerThenTerm = SplitBlockAndInsertIfThen(IsNewPage, ThenTerm, false);
            IRBuilder<> InnerBuilder(InnerThenTerm);
            InnerBuilder.CreateCall(MarkFunc, {AddrInt});
            InnerBuilder.CreateStore(CurPage, LocalCacheVar);

            Changed = true;
            UVM_DBGS() << "   [+] Instrumented: " << *Inst << "\n";
        }
    }

    // 6. Extract automated metadata expressions for all user NVVM kernels
    std::vector<KernelMetadataBytecode> AllMetadata;
    for (Function *OrigF : UserFunctions) {
        if (isNvvmKernel(*OrigF)) {
            auto &LI = FAM.getResult<LoopAnalysis>(*OrigF);
            auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(*OrigF);
            KernelMetadataBytecode meta = extractKernelMetadata(*OrigF, SE, LI, M.getDataLayout());
            AllMetadata.push_back(meta);
            UVM_DBGS() << "[UvmPass] Extracted metadata for " << OrigF->getName()
                       << " (ranges: " << meta.num_ranges << ", bytecode len: "
                       << meta.bytecode.size() << ")\n";
        }
    }
    emitModuleMetadataTable(M, AllMetadata);

    UVM_DBGS() << "[UvmPass] Finished processing module: " << M.getSourceFileName() << "\n";
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
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "uvm-tracking") {
                        MPM.addPass(UvmTrackingPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
