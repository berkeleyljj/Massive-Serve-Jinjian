## DiskANN-build quick setup (Debian 12)

Install base build tools and dependencies:
```bash
sudo apt install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

Install Intel MKL (runtime):
```bash
  wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/79153e0f-74d7-45af-b8c2-258941adf58a/intel-onemkl-2025.0.0.940.sh
  sudo sh intel-onemkl-2025.0.0.940.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
  source /opt/intel/oneapi/setvars.sh
```
