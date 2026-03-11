; ModuleID = '../test_kernel.cu'
source_filename = "../test_kernel.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%printf_args = type { i32 }
%printf_args.0 = type { i32 }
%printf_args.1 = type { i32 }

@.str = private unnamed_addr constant [26 x i8] c"Hello from GPU thread %d\0A\00", align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@.str1 = private unnamed_addr constant [34 x i8] c"Stride: Hello from GPU thread %d\0A\00", align 1
@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@.str2 = private unnamed_addr constant [18 x i8] c"TID: %d concluded\00", align 1

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local ptx_kernel void @_Z8myKernelv() #0 {
  %1 = alloca %printf_args, align 8
  %2 = call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = getelementptr inbounds nuw %printf_args, ptr %1, i32 0, i32 0
  store i32 %2, ptr %3, align 4
  %4 = call i32 @vprintf(ptr @.str, ptr %1)
  ret void
}

declare i32 @vprintf(ptr, ptr)

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local ptx_kernel void @_Z13stride_accessPii(ptr noundef %0, i32 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca %printf_args.0, align 8
  %6 = alloca i32, align 4
  %7 = alloca %printf_args.1, align 8
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %8 = call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %9 = getelementptr inbounds nuw %printf_args.0, ptr %5, i32 0, i32 0
  store i32 %8, ptr %9, align 4
  %10 = call i32 @vprintf(ptr @.str1, ptr %5)
  %11 = call noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %12 = call noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %13 = mul i32 %11, %12
  %14 = call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %15 = add i32 %13, %14
  store i32 %15, ptr %6, align 4
  %16 = load i32, ptr %6, align 4
  %17 = load i32, ptr %4, align 4
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %26

19:                                               ; preds = %2
  %20 = load i32, ptr %6, align 4
  %21 = load ptr, ptr %3, align 8
  %22 = load i32, ptr %6, align 4
  %23 = mul nsw i32 %22, 1024
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds i32, ptr %21, i64 %24
  store i32 %20, ptr %25, align 4
  br label %26

26:                                               ; preds = %19, %2
  %27 = load i32, ptr %6, align 4
  %28 = getelementptr inbounds nuw %printf_args.1, ptr %7, i32 0, i32 0
  store i32 %27, ptr %28, align 4
  %29 = call i32 @vprintf(ptr @.str2, ptr %7)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_89" "target-features"="+ptx78,+sm_89" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4, !5}
!llvm.ident = !{!6, !7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 8]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @_Z8myKernelv}
!5 = !{ptr @_Z13stride_accessPii}
!6 = !{!"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"}
!7 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
