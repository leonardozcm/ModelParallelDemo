# ModelParallelDemo
This repo restores the exploration of pytorch and tf model parallel on intel cpu

# 8.15
```
❯ python model.py
/home/chriskafka/anaconda3/envs/py37/lib/python3.7/site-packages/torchvision/models/_utils.py:209: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
  f"The parameter '{pretrained_param}' is deprecated since 0.13 and will be removed in 0.15, "
/home/chriskafka/anaconda3/envs/py37/lib/python3.7/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
2022-08-15 01:20:10,651 INFO services.py:1476 -- View the Ray dashboard at http://127.0.0.1:8265
E0815 01:20:14.811084000   12838 socket_utils_common_posix.cc:223] check for SO_REUSEPORT: {"created":"@1660497614.811072600","description":"Protocol not available","errno":92,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":202,"os_error":"Protocol not available","syscall":"getsockopt(SO_REUSEPORT)"}
(raylet) E0815 01:20:15.910932900   13440 socket_utils_common_posix.cc:223] check for SO_REUSEPORT: {"created":"@1660497615.910921100","description":"Protocol not available","errno":92,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":202,"os_error":"Protocol not available","syscall":"getsockopt(SO_REUSEPORT)"}
(raylet) E0815 01:20:15.918901200   13442 socket_utils_common_posix.cc:223] check for SO_REUSEPORT: {"created":"@1660497615.918886200","description":"Protocol not available","errno":92,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":202,"os_error":"Protocol not available","syscall":"getsockopt(SO_REUSEPORT)"}
(raylet) E0815 01:20:15.926445200   13441 socket_utils_common_posix.cc:223] check for SO_REUSEPORT: {"created":"@1660497615.926428400","description":"Protocol not available","errno":92,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":202,"os_error":"Protocol not available","syscall":"getsockopt(SO_REUSEPORT)"}
(raylet) E0815 01:20:15.926523700   13443 socket_utils_common_posix.cc:223] check for SO_REUSEPORT: {"created":"@1660497615.926507500","description":"Protocol not available","errno":92,"file":"external/com_github_grpc_grpc/src/core/lib/iomgr/socket_utils_common_posix.cc","file_line":202,"os_error":"Protocol not available","syscall":"getsockopt(SO_REUSEPORT)"}
net output size:
torch.Size([8, 16384])
torch.Size([8, 16384])
torch.Size([8, 16384])
torch.Size([8, 16384])
```