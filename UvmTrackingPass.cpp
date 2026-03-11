#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Metadata.h" // Ensure this is included
                              //
using namespace llvm;

// 1. Explicitly define as a Module Pass
class UvmTrackingPass : public PassInfoMixin<UvmTrackingPass> {
public:
    /*PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        auto &Ctx = M.getContext();
        FunctionCallee MarkFunc = M.getOrInsertFunction("MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
        GlobalVariable *CacheVar = M.getGlobalVariable("last_page_cache");

        if (!CacheVar) return PreservedAnalyses::all();

        bool Changed = false;
        for (auto &F : M) {
            if (F.isDeclaration() || F.getName() == "MarkAccess") continue;

            for (auto &BB : F) {
                // Use a stable iterator because we are splitting blocks
                for (auto I = BB.begin(), E = BB.end(); I != E; ++I) {
                    Instruction *Inst = &*I;
                    Value *Ptr = nullptr;

                    if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                        if (LI->getPointerAddressSpace() == 1) Ptr = LI->getPointerOperand();
                    } else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
                        if (SI->getPointerAddressSpace() == 1) Ptr = SI->getPointerOperand();
                    }

                    if (Ptr) {
                        IRBuilder<> Builder(Inst);
                        Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
                        Value *CurPage = Builder.CreateLShr(AddrInt, 12);
                        Value *LastPage = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
                        Value *IsNewPage = Builder.CreateICmpNE(CurPage, LastPage);

                        // Splitting the block safely
                        Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
                        Builder.SetInsertPoint(ThenTerm);
                        Builder.CreateCall(MarkFunc, {AddrInt});
                        Builder.CreateStore(CurPage, CacheVar);
                        
                        Changed = true;
                        // Move iterator past the new instructions
                        break; 
                    }
                }
            }
        }
        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
    */

/*
PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    auto &Ctx = M.getContext();
    FunctionCallee MarkFunc = M.getOrInsertFunction("MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
    GlobalVariable *CacheVar = M.getGlobalVariable("last_page_cache");

    if (!CacheVar) {
        CacheVar = new GlobalVariable(M, Type::getInt64Ty(Ctx), false, 
                                     GlobalValue::ExternalLinkage, nullptr, 
                                     "last_page_cache", nullptr, 
                                     GlobalValue::NotThreadLocal, 1);
    }

    // 1. Create a metadata node to identify our injected instructions
    MDNode *Node = MDNode::get(Ctx, MDString::get(Ctx, "nosanitize"));

    bool Changed = false;
    for (auto &F : M) {
        // Skip MarkAccess and declarations
        if (F.isDeclaration() || F.getName().contains("MarkAccess")) continue;

        for (auto &BB : F) {
            for (auto I = BB.begin(), E = BB.end(); I != E; ++I) {
                Instruction *Inst = &*I;

                // 2. THE CHECK: If this instruction is already tagged, skip it!
                if (Inst->getMetadata("nosanitize")) continue;

                Value *Ptr = nullptr;
                if (auto *LI = dyn_cast<LoadInst>(Inst)) Ptr = LI->getPointerOperand();
                else if (auto *SI = dyn_cast<StoreInst>(Inst)) Ptr = SI->getPointerOperand();

                // Check Address Space (Generic 0 or Global 1)
                if (Ptr && Ptr->getType()->isPointerTy()) {
                    unsigned AS = Ptr->getType()->getPointerAddressSpace();
                    if (AS <= 1) {
                        
                        IRBuilder<> Builder(Inst);
                        
                        // 3. THE TAGGING: Attach metadata to every NEW instruction we create
                        Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
                        cast<Instruction>(AddrInt)->setMetadata("nosanitize", Node);

                        Value *CurPage = Builder.CreateLShr(AddrInt, 12);
                        cast<Instruction>(CurPage)->setMetadata("nosanitize", Node);

                        auto *LastPageLoad = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
                        LastPageLoad->setMetadata("nosanitize", Node);

                        auto *IsNewPage = Builder.CreateICmpNE(CurPage, LastPageLoad);
                        cast<Instruction>(IsNewPage)->setMetadata("nosanitize", Node);

                        Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
                        
                        Builder.SetInsertPoint(ThenTerm);
                        auto *GpuCall = Builder.CreateCall(MarkFunc, {AddrInt});
                        GpuCall->setMetadata("nosanitize", Node);

                        auto *CacheStore = Builder.CreateStore(CurPage, CacheVar);
                        CacheStore->setMetadata("nosanitize", Node);

                        errs() << "[UvmPass] Instrumented and Tagged: " << *Inst << "\n";
                        Changed = true;

                        // Break the inner loop for this block because SplitBlockAndInsertIfThen
                        // just mangled the block structure. Move to the next block.
                        break; 
                    }
                }
            }
        }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        auto &Ctx = M.getContext();
        FunctionCallee MarkFunc = M.getOrInsertFunction("MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
        GlobalVariable *CacheVar = M.getGlobalVariable("last_page_cache");

        if (!CacheVar) {
            errs() << "[UvmPass] Error: last_page_cache not found in Module!\n";

            CacheVar = new GlobalVariable(M, 
                    Type::getInt64Ty(Ctx), 
                    false, 
                    GlobalValue::ExternalLinkage, 
                    nullptr, 
                    "last_page_cache",
                    nullptr,
                    GlobalValue::NotThreadLocal,
                    1); // Address Space 1 (Global Memory)

            errs() << "[UvmPass] Created external declaration for last_page_cache\n";
        }

        bool Changed = false;
        for (auto &F : M) {
            // Log every function we encounter
            errs() << "[UvmPass] Analyzing function: " << F.getName() << "\n";

            if (F.isDeclaration() || F.getName() == "MarkAccess") continue;

            for (auto &BB : F) {
                for (auto I = BB.begin(), E = BB.end(); I != E; ++I) {
                    Instruction *Inst = &*I;
                    // 1. At the start of your instruction loop, check for the tag
                    if (Inst->getMetadata("nosanitize")) {
                        continue; 
                    }
                    Value *Ptr = nullptr;
                    std::string OpType = "";

                    if (auto *LI = dyn_cast<LoadInst>(Inst)) {
                        if (LI->getPointerAddressSpace() == 1) {
                            Ptr = LI->getPointerOperand();
                            OpType = "Load";
                        }
                    } else if (auto *SI = dyn_cast<StoreInst>(Inst)) {
                        if (SI->getPointerAddressSpace() == 1) {
                            Ptr = SI->getPointerOperand();
                            OpType = "Store";
                        }
                    }

                    if (Ptr == CacheVar) continue;
                    // ... (your existing logic to find Load/Store) ...

                    if (Ptr) {
                        // 2. Create the metadata node
                        LLVMContext &C = M.getContext();
                        MDNode *Node = MDNode::get(C, MDString::get(C, "nosanitize"));

                        IRBuilder<> Builder(Inst);

                        // 3. For every instruction you create, attach the metadata
                        Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
                        cast<Instruction>(AddrInt)->setMetadata("nosanitize", Node);

                        Value *CurPage = Builder.CreateLShr(AddrInt, 12);
                        cast<Instruction>(CurPage)->setMetadata("nosanitize", Node);

                        auto *LastPageLoad = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
                        LastPageLoad->setMetadata("nosanitize", Node);

                        auto *IsNewPage = Builder.CreateICmpNE(CurPage, LastPageLoad);
                        cast<Instruction>(IsNewPage)->setMetadata("nosanitize", Node);

                        Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
                        // Note: SplitBlockAndInsertIfThen creates a branch, it usually doesn't need tagging 
                        // but the instructions inside the 'then' block definitely do.

                        Builder.SetInsertPoint(ThenTerm);
                        auto *GpuCall = Builder.CreateCall(MarkFunc, {AddrInt});
                        GpuCall->setMetadata("nosanitize", Node);

                        auto *CacheStore = Builder.CreateStore(CurPage, CacheVar);
                        CacheStore->setMetadata("nosanitize", Node);

                        errs() << "     [+] Instrumentation injected and tagged.\n";
                        Changed = true;

                        // 4. IMPORTANT: Skip past the original instruction we just instrumented 
                        // to avoid re-scanning the same block repeatedly in some LLVM versions.
                        // (You already have a 'break' in your current code, keep it!)
                        break; 
                    }
                    if (Ptr) {
                        errs() << "  -> Found " << OpType << " in AS(1): " << *Inst << "\n";

                        IRBuilder<> Builder(Inst);
                        Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
                        Value *CurPage = Builder.CreateLShr(AddrInt, 12);
                        Value *LastPage = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
                        Value *IsNewPage = Builder.CreateICmpNE(CurPage, LastPage);

                        Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
                        Builder.SetInsertPoint(ThenTerm);
                        Builder.CreateCall(MarkFunc, {AddrInt});
                        Builder.CreateStore(CurPage, CacheVar);

                        errs() << "     [+] Instrumentation injected.\n";
                        Changed = true;
                        break; 
                    }
                }
            }
        }
        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }*/

PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
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


    /*
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        auto &Ctx = M.getContext();
        FunctionCallee MarkFunc = M.getOrInsertFunction("MarkAccess", Type::getVoidTy(Ctx), Type::getInt64Ty(Ctx));
        GlobalVariable *CacheVar = M.getGlobalVariable("last_page_cache");

        // 1. Declare CacheVar if missing (as we did before)
        if (!CacheVar) {
            CacheVar = new GlobalVariable(M, Type::getInt64Ty(Ctx), false, 
                    GlobalValue::ExternalLinkage, nullptr, 
                    "last_page_cache", nullptr, 
                    GlobalValue::NotThreadLocal, 1);
        }

        // 2. WORKLIST: Collect target instructions first
        std::vector<Instruction*> Worklist;
        for (auto &F : M) {
            if (F.isDeclaration() || F.getName().contains("MarkAccess")) continue;

            for (auto &BB : F) {
                for (auto &Inst : BB) {
                    Value *Ptr = nullptr;
                    if (auto *LI = dyn_cast<LoadInst>(&Inst)) Ptr = LI->getPointerOperand();
                    else if (auto *SI = dyn_cast<StoreInst>(&Inst)) Ptr = SI->getPointerOperand();

                    // Skip if it's an internal cache update
                    if (Ptr && Ptr != CacheVar && Ptr->getType()->isPointerTy()) {
                        unsigned AS = Ptr->getType()->getPointerAddressSpace();
                        if (AS == 0 || AS == 1) {
                            Worklist.push_back(&Inst);
                        }
                    }
                }
            }
        }

        // 3. INSTRUMENTATION: Process the worklist (Safe from recursion)
        errs() << "[UvmPass] Found " << Worklist.size() << " instructions to instrument.\n";
        for (Instruction *Inst : Worklist) {
            Value *Ptr = (isa<LoadInst>(Inst)) ? cast<LoadInst>(Inst)->getPointerOperand() 
                : cast<StoreInst>(Inst)->getPointerOperand();

            IRBuilder<> Builder(Inst);
            Value *AddrInt = Builder.CreatePtrToInt(Ptr, Builder.getInt64Ty());
            Value *CurPage = Builder.CreateLShr(AddrInt, 12);
            Value *LastPage = Builder.CreateLoad(Builder.getInt64Ty(), CacheVar);
            Value *IsNewPage = Builder.CreateICmpNE(CurPage, LastPage);

            Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsNewPage, Inst, false);
            Builder.SetInsertPoint(ThenTerm);
            Builder.CreateCall(MarkFunc, {AddrInt});
            Builder.CreateStore(CurPage, CacheVar);
        }

        return Worklist.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
    }*/
};

// 2. Simplified Registration Logic
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "UvmTrackingPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "uvm-track") {
                        MPM.addPass(UvmTrackingPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
