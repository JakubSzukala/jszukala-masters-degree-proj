# jszukala-masters-degree-proj

Ideas for the masters project:
1. Visual deffect detection on PCBs with neural networks
    * NN would analyze a photo of some small fragment of PCB and classify detected joints to a class, for example: correct joint, tombstoned component, cold weld etc
    * Complex problem
    * Can be reduced to detection of specific class of deffects (tombstones for example)
    * Not really much of data available, so some data gathering will be necessary
2. Neural network routing a PCB
    * NN would solve a problem of routing traces / connecting components on PCB, in other words, it would be an auto routing algorithm that would utilize a NN
    * It would be pretty nice alternative to classical auto routing algorithms that usually are not very respected and not widely used by professionals
    * Complex problem, with big search space
    * There are similar attempts and projects ([1](https://www.deeppcb.ai/), [2](https://arxiv.org/pdf/2006.13607.pdf), [3](https://dspace.mit.edu/bitstream/handle/1721.1/129238/1227515700-MIT.pdf?sequence=1&isAllowed=y))
3. Analysis of various techniques excersised by athletes
    * Analysis of runners techniques by gathering data from some on body accelerometers, detection of incorrect running technique that may lead to injuries
    * Maybe it could be determined in comparission to experienced runners
    * Maybe analysis could be performed live, with use of some set of small sensors and microcontroller with trained model on Tensor Flow Lite
    * Problem is similar to [magic wand](https://codelabs.developers.google.com/magicwand) but more complex
    * Data set could be created, but from short research I found some [data set](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5426356/) that may be suitable
4. Conversion of satelite images into urbanistic analysis
    * It boils down to segmentation task
    * It would be nice for architects, that often need to perform tedious tasks of various analysis on satelite images and other maps
5. Power optimization for Linux system / laptop battery with use of some NN
    * Alternative to [TLP](https://linrunner.de/tlp/index.html)
    * Could adjust power settings depending on tasks executed, power supply etc
    * Data will have to be gathered
    * Maybe as a use case, we could perform some artificial calculation task, which algorithm would have to optimize in terms of power consumption, with minimal impact on performance
    * Maybe it would be more suitable to some IoT remote devices running Linux like baseboard with [jetson computing module](https://antmicro.com/platforms/open-source-jetson-baseboard/), [BeagleBone](https://beagleboard.org/bone) or even [RPI](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
    * In Linux there is [quite a few ways](https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-devices-power) of devices control.
6. Battery life and condition estimation with NN
    * [Previous work](http://li.mit.edu/A/Papers/22/Hsu22XiongAE.pdf)
    * Data set could be created by discharging battery, or some [example](https://kilthub.cmu.edu/articles/dataset/eVTOL_Battery_Dataset/14226830) data set can be used
7. Visual inspection of plants' health
8. Any other intresting topic, I am intrested in bunch of things and open minded :)
