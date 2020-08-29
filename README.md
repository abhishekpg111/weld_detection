# weld_detection
<p align="justify">    
Welding is a fundamental and important process in manufacturing. The shortage of skilled welding labour and the pressure to increase production, cut costs and increase workplace safety has increased demand for robotic welding. Currently, industrial arc welding robots are programmed through ‘teach and playback’ methods by human operators. It can take a significant amount of time and expense to programme paths and refine welding parameters for each new part. This setup time can be justified in mass production, however for low to medium volume manufacturing or repair and maintenance work it is often quicker and cheaper to weld the parts manually. In order to solve this problem, sensor needs to be integrated with the robot. Computer vision system can be used to detect the weld seams and
provide a path to the robot to weld the parts automatically. However, the welding environment presents unique challenges for computer vison. These challenges include poor contrast, reflections from metallic surfaces, and imperfections on the work piece such as rust, mill scale and scratches which are not consistent from part to part. This project is developed for detectin the weld joint using the image processing techniques. It contains two seperate packages for detecting butt weld joints and a generalized code for detecting all types of weld joints using a line growing algorithm. 
 
</p>
 
## Development Environment
- __Ubuntu 16.04.2__
- __python 2.7__
- __OpenCv 3.3.1__
