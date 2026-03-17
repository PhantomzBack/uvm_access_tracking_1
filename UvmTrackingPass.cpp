#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Metadata.h"
                              
using namespace llvm;

class UvmTrackingPass : public PassInfoMixin<UvmTrackingPass> {
public:
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        printf("[UvmPass] Starting UVM tracking pass in source file %s\n", M.getSourceFileName().c_str());
        if (M.getTargetTriple().find("nvptx") == std::string::npos)
            return PreservedAnalyses::all();
        printf("[UvmPass] Running on target triple: '%s'\n", M.getTargetTriple().c_str());

        if (M.getSourceFileName().find("libMarkAccess") != std::string::npos){
            printf("[UvmPass] Skipping module '%s'\n", M.getSourceFileName().c_str());
            return PreservedAnalyses::all();
        }
        printf("[UvmPass] Running on module: '%s'\n", M.getSourceFileName().c_str());
        auto &Ctx = M.getContext();
        FunctionCallee MarkFunc = M.getOrInsertFunction("MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
        GlobalVariable *CacheVar = M.getGlobalVariable("last_page_cache");

        if (!CacheVar) {
            CacheVar = new GlobalVariable(M, Type::getInt64Ty(Ctx), false, 
                                        GlobalValue::ExternalLinkage, nullptr, 
                                        "last_page_cache", nullptr, 
                                        GlobalValue::NotThreadLocal, 1);
        }

        // 1. PHASE ONE: Collect instructions to instrument
        std::vector<Instruction*> Targets;
        for (auto &F : M) {
            if (!F.hasFnAttribute("target-features"))
                continue;
            if (!F.getFnAttribute("target-features").getValueAsString().contains("ptx"))
                continue;
            if (F.isDeclaration() || F.getName().contains("MarkAccess")) continue;

            for (auto &BB : F) {
                for (auto &Inst : BB) {
                    Value *Ptr = nullptr;
                    if (auto *LI = dyn_cast<LoadInst>(&Inst)) Ptr = LI->getPointerOperand();
                    else if (auto *SI = dyn_cast<StoreInst>(&Inst)) Ptr = SI->getPointerOperand();

                    // Skip internal cache updates
                    if (Ptr && Ptr != CacheVar && Ptr->getType()->isPointerTy()) {
                        unsigned AS = Ptr->getType()->getPointerAddressSpace();
                        // Instrument both Generic (0) and Global (1)
                        if (AS <= 1) {
                            Targets.push_back(&Inst);
                        }
                    }
                }
            }
        }

        // 2. PHASE TWO: Process the Worklist
        errs() << "[UvmPass] Found " << Targets.size() << " target instructions.\n";
        for (Instruction *Inst : Targets) {
            Value *Ptr = (isa<LoadInst>(Inst)) ? cast<LoadInst>(Inst)->getPointerOperand() 
                                            : cast<StoreInst>(Inst)->getPointerOperand();

            IRBuilder<> Builder(Inst);
            Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
            Value *CurPage = Builder.CreateLShr(AddrInt, 12);
            Value *LastPage = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
            Value *IsNewPage = Builder.CreateICmpNE(CurPage, LastPage);

            // This is safe now because we aren't iterating over these blocks anymore
            Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
            Builder.SetInsertPoint(ThenTerm);
            Builder.CreateCall(MarkFunc, {AddrInt});
            Builder.CreateStore(CurPage, CacheVar);
            
            errs() << "   [+] Instrumented: " << *Inst << "\n";
        }

        return Targets.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
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
