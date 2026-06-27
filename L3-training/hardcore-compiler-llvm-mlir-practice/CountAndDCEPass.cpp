// CountAndDCEPass.cpp —— 基于新 PassManager 的 FunctionPass 插件
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

// 一个 FunctionPass：统计指令数 + 简单死代码消除（DCE）
struct CountAndDCEPass : PassInfoMixin<CountAndDCEPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    unsigned before = 0;
    for (auto &BB : F)
      before += BB.size();
    errs() << "[CountAndDCE] 函数 '" << F.getName()
           << "' 改写前指令数 = " << before << "\n";

    // —— 死代码消除：遍历每条指令，删除「无副作用 + 无使用者」的指令 ——
    bool changed = false;
    SmallVector<Instruction *, 16> dead;
    for (auto &BB : F)
      for (auto &I : BB)
        // 不是终结指令、不产生副作用、且结果无人使用 => 死代码
        if (!I.isTerminator() && !I.mayHaveSideEffects() && I.use_empty())
          dead.push_back(&I);

    for (Instruction *I : dead) {
      I->eraseFromParent();
      changed = true;
    }

    unsigned after = 0;
    for (auto &BB : F)
      after += BB.size();
    errs() << "[CountAndDCE] 函数 '" << F.getName()
           << "' 改写后指令数 = " << after
           << "（消除 " << (before - after) << " 条死代码）\n";

    // 改了 CFG 之外的内容，但保留了 CFG 分析结果
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  // 让该 Pass 对 optnone 函数也生效（演示用）
  static bool isRequired() { return true; }
};

} // namespace

// —— 注册为新 PassManager 插件，供 opt -passes=count-and-dce 调用 ——
llvm::PassPluginLibraryInfo getCountAndDCEPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountAndDCE", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "count-and-dce") {
                    FPM.addPass(CountAndDCEPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// opt 加载插件时调用的入口符号
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getCountAndDCEPluginInfo();
}
