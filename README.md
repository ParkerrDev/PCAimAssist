# PCAimAssist

Fork of https://github.com/RootKit-Org/AI-Aimbot

## Changes and Optimizations
- Uses OBS studio websocket to capture game windows directly.
- Improved targeting logic and enhancements for third person shooters without interference.
- Greater image input 640x640 instead of 320x320.
- Wider screen view, not only is the image greater but it captures more screen.
- Automatic firing capabilities.
- Fixes previous issues with moving the window.
- More customizable and configurable.


## 🧰 Requirements
- Nvidia RTX 980 🆙, higher or equivalent
- And one of the following:
  - Nvidia CUDA Toolkit 11.8 [DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

## 🚀 Pre-setup Steps
1. Download and Unzip the AI Aimbot and stash the folder somewhere handy 🗂️.
2. Ensure you've got Python installed (like a pet python 🐍) – grab version 3.11 [HERE](https://www.python.org/downloads/release/python-3116/).
   - 🛑 Facing a `python is not recognized...` error? [WATCH THIS!](https://youtu.be/E2HvWhhAW0g)
   - 🛑 Is it a `pip is not recognized...` error? [WATCH THIS!](https://youtu.be/zWYvRS7DtOg)
3. Fire up `PowerShell` or `Command Prompt` on Windows 🔍.
4. To install `PyTorch`, select the appropriate command based on your GPU.
    - Nvidia `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118`
    - AMD or CPU `pip install torch torchvision torchaudio`
5. 📦 Run the command below to install the required Open Source packages:
```
pip install -r requirements.txt
```

## 🔌 How to Run
Follow these sparkly steps to get your TensorRT ready for action! 🛠️✨

1. **Introduction (for original repo)** 🎬
   Watch the TensorRT section of the setup for the original repo --> [video 🎥](https://www.youtube.com/watch?v=uniL5yR7y0M&ab_channel=RootKit) before you begin. It's loaded with useful tips!

2. **Set Environment Variables** 🌱
  - Press Win + R, type sysdm.cpl, and press Enter.
  - Go to the Advanced tab and click on Environment Variables.
  - Under System Variables, scroll down and find Path, then click Edit.
  - Add the Required Paths
  - Click New and add the following one by one:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
  ```
4. **Get Support If You're Stumped** 🤔
   If you ever feel lost, you can always `@Wonder` your questions in our [Discord 💬](https://discord.gg/rootkitorg). Wonder is here to help!

5. **Install Cupy**
    Run the following `pip install cupy-cuda11x`

6. **CUDNN Installation** 🧩
   Click to install [CUDNN 📥](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). You'll need a Nvidia account to proceed. Don't worry it's free.

7. **Unzip and Relocate** 📁➡️
   Open the .zip CuDNN file and move all the folders/files to where the CUDA Toolkit is on your machine, usually at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.

8. **Get TensorRT 8.6 GA** 🔽
   Fetch [`TensorRT 8.6 GA 🛒`](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

9. **Unzip and Relocate** 📁➡️
   Open the .zip TensorRT file and move all the folders/files to where the CUDA Toolkit is on your machine, usually at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.

10. **Python TensorRT Installation** 🎡
   Once you have all the files copied over, you should have a folder at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python`. If you do, good, then run the following command to install TensorRT in python.
   ```
   pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
   ```
    🚨 If the following steps didn't work, don't stress out! 😅 The labeling of the files corresponds with the Python version you have installed on your machine. We're not looking for the 'lean' or 'dispatch' versions. 🔍 Just locate the correct file and replace the path with your new one. 🔄 You've got this! 💪    

11. **Create the onnx file** 🏃‍♂️💻
   ```
   python .\export.py --weights ./yolov5s.pt --include onnx --half --imgsz 640 640 --device 0
   ```

12. **Build the .engine file on your system** 🤖
  ```
  trtexec --onnx=yolov5x.onnx --saveEngine=yolov5s.engine --fp16
  ```

  ## Now just run it!!
   ```
   python ./main.py
   ```

If you've followed these steps, you should be asked which window to use and it should be displaying a visual of the current game! ⚙️🚀

