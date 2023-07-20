# jszukala-masters-degree-proj
Wheat head detection on GWHD_2021 dataset

PUT masters degree project. Work in progress.

#### Current state of the project
Currently I was able to quite successfully train stock YOLOv7 networks to perform the task. Current work is focused on custom implementation of YOLOv7 that allows for far greater flexibility and my intention is to use this flexibility to gain an edge over the stock YOLOv7 results with techniques like image augmentation and pseudo labelling.

#### TODOS
Very general list of TODOs:
- [x] Perform data exploration
- [x] Cleanup data
- [x] Create cloud based training
- [x] Perform training on ["stock" YOLOv7](https://github.com/WongKinYiu/yolov7)
    - [x] Train regular YOLOv7
    - [x] Train largest model YOLOv7-e6e
- [ ] Implement custom training loop using Chris Hughes's from Microsoft YOLOv7 [implementation](https://github.com/Chris-hughes10/Yolov7-training)
    - [x] Implement example training loop
    - [x] Overfit the model to see if the training procedure works as intended
    - [ ] Implement and track the same metrics that are by default used in stock YOLOv7 for more accurate comparission
    - [ ] Add performance enhancements to the base training:
        - [ ] Image data augmentations
        - [ ] Pseudo labelling
        - [ ] ...
- [ ] Make comparission between my results and results of the competition winners
- [ ] Create a heat map describing wheat density on geo - annotated wheat image data