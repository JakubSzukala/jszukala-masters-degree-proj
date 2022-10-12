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
8. Reinforcement learning algorithm that would control a camera while live spectating a game like Star Craft 2
    * Spectating is hard to automate and usually human operators need quite some experience and game knowledge to skillfully control camera during live play, to catch the state of the game, higlight important plays and avoid viewer confusion
    * There is already prepared very nice [environment](https://github.com/deepmind/pysc2), by Deepmind, that exposes the Star Craft 2 both observations and actions
    * There is quite big [data base of replays](https://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1-fix.zip) that are suitable for that particular environment, we are only constrained by [available versions of the game that have those RL extensions built in](https://github.com/Blizzard/s2client-proto#downloads) and we can also perform training on some bots game instead of replay.
    * The environment currently exposes only orthogonal camera render from what I've seen and for spectating there is always used perspective camera, so that functionality would have to be added
    * The task of the algorithm would be to position camera is such a way to catch as many relevant events as possible (in its field of view) and each game event would have lets say position which could be checked whether camera catches it or not and reward value for observing it. The higher the value, the more important the event
9. Any other intresting topic, I am intrested in bunch of things and open minded :)

