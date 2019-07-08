### BanditBox: A reproducible and portable GPU and TPU accelerated machine learning container 

GPU Accelerated computing container for machine learning and NLP, that are reproducible across environments.

-----------------------------------------------------------

### BanditBox Features ###

* [Contextual Bandits](https://contextual-bandits.readthedocs.io/en/latest/#installation)
  Contains working examples in ` apps/* ` of various gradient policy networks; Framework handles contextual multi-armed bandits. Adaptations from multi-armed bandits strategies you can use: Upper-confidence Bound, Thompson Sampling, Epsilon Greedy, Adaptive Greedy (Adaptive-Greedy algorithm shows a lot of promise!!), Explore-Then-Exploit.
  
* [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki)
Vowpal Wabbit (VW) is a high performance machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning. 
The contextual bandit learning algorithms in VW consist of two broad classes. The first class consists of settings where the maximum number of actions is known ahead of time, and the semantics of these actions stay fixed across examples. A more advanced setting allows potentially changing semantics per example.

### GPU/TPU Machine Learning & Distributed Pipelines for ultra fast inference

* [Dask Distributed](https://dask.org/) 
* [TensorFlow for GPU v1.13.1](https://www.tensorflow.org/install/gpu)
* [TensorBoard](https://www.datacamp.com/community/tutorials/tensorboard-tutorial)
* [TensorFlowServing Python API](https://www.tensorflow.org/tfx/guide/serving)
* [NVIDIA TensorRT inference accelerator and CUDA 10](https://developer.nvidia.com/tensorrt)
* [PyCUDA 2019](https://mathema.tician.de/software/pycuda/)
* [CuPy:latest](https://cupy.chainer.org/)
* Ubuntu 18.04 so you can 'nix your way through the cmd line!
* cuDNN7.4.1.5 for deeep learning in CNN's

### How else do you plan to serve a model to production?
* [TensorFlow-serving-api](https://www.tensorflow.org/tfx/guide/serving)

### Good to know
* Hot Reloading: code updates will automatically update in container from /apps folder.
* TensorBoard is on localhost:6006 and GPU enabled Jupyter is on localhost:8888.
* Python 3.6.7 (Stable & Secure)
* Only Tesla Pascal and Turing GPU Architecture are supported 

-------------------------------------------------------------


### Before you begin (This might be optional) ###

Link to nvidia-docker2 install: [Tutorial](https://medium.com/@sh.tsang/docker-tutorial-5-nvidia-docker-2-0-installation-in-ubuntu-18-04-cb80f17cac65)

You must install nvidia-docker2 and all it's deps first, assuming that is done, run:


 ` sudo apt-get install nvidia-docker2 `
 
 ` sudo pkill -SIGHUP dockerd `
 
 ` sudo systemctl daemon-reload `
 
 ` sudo systemctl restart docker `
 

How to run this container:


### Step 1 ###

` docker build -t <container name> . `  < note the . after <container name>


### Step 2 ###

Run the image, mount the volumes for Jupyter and app folder for your fav IDE, and finally the expose ports `8888` for TF1, and `6006` for TensorBoard.


` docker run --rm -it --runtime=nvidia --user $(id -u):$(id -g) --group-add container_user --group-add sudo -v "${PWD}:/apps" -v $(pwd):/tf/notebooks  -p 8888:8888 -p 0.0.0.0:6006:6006  <container name> `


### Step 3: Check to make sure GPU drivers and CUDA is running ###

- Exec into the container and check if your GPU is registering in the container and CUDA is working:

- Get the container id:

` docker ps `

- Exec into container:

` docker exec -u root -t -i <container id> /bin/bash `

- Check if NVIDIA GPU DRIVERS have container access:

` nvidia-smi `

- Check if CUDA is working:

` nvcc -V `


### Step 4: How to launch TensorBoard ###

(It helps to use multiple tabs in cmd line, as you have to leave at least 1 tab open for TensorBoard@:6006)

- Demonstrates the functionality of TensorBoard dashboard


- Exec into container if you haven't, as shown above:


- Get the `<container id>`:
 

` docker ps `


` docker exec -u root -t -i <container id> /bin/bash `


- Then run in cmd line:


` tensorboard --logdir=//tmp/tensorflow/mnist/logs `


- Type in: ` cd / ` to get root.

Then cd into the folder that hot reloads code from your local folder/fav IDE at: `/apps/apps/gpu_benchmarks` and run:


` python tensorboard.py `


- Go to the browser and navigate to: ` localhost:6006 `



### Step 5: Run tests to prove container based GPU perf ###

- Demonstrate GPU vs CPU performance:

- Exec into the container if you haven't, and cd over to /tf/notebooks/apps/gpu_benchmarks and run:

- CPU Perf:

` python benchmark.py cpu 10000 `

- CPU perf should return something like this:

`Shape: (10000, 10000) Device: /cpu:0
Time taken: 0:00:03.934996`

- GPU perf:

` python benchmark.py gpu 10000 `

- GPU perf should return something like this:

`Shape: (10000, 10000) Device: /gpu:0
Time taken: 0:00:01.032577`


--------------------------------------------------


### Known conflicts with nvidia-docker and Ubuntu ###

AppArmor on Ubuntu has sec issues, so remove docker from it on your local box, (it does not hurt security on your computer):

` sudo aa-remove-unknown `

--------------------------------------------------

If building impactful data science tools is important to you or your business, please get in touch.

#### Contact
Email: joehoeller@gmail.com




