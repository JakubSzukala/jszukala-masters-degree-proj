// Constants
const IMG_PER_PAGE = 20;

let imageToBBoxesMap;
let filteredImages = [];
let currentLowerIndex = 0;
let currentUpperIndex = IMG_PER_PAGE;


async function parseBBoxCSV(fileUrl, hasHeader = true) {
    try {
        const dictionary = new Map();
        const response = await fetch(fileUrl);
        const csvContent = await response.text();
        const rows = csvContent.split('\n');

        if (hasHeader) {
            rows.shift();
        }

        for (let i = 0; i < rows.length; i++) {
            if (rows[i] == '') {
                continue;
            }

            [image_name, bboxesString] = rows[i].split(',');
            if (bboxesString == 'no_box') {
                continue;
            }
            filteredImages.length
            bboxesList = bboxesString.split(';');

            for (let j = 0; j < bboxesList.length; j++) {
                bboxesList[j] = bboxesList[j].split(' ').map(str => parseInt(str));
            }
            if (dictionary.has(image_name)) {
                const existingBboxesList = dictionary.get(image_name);
                for (let i = 0; i < bboxesList.length; i++) {
                    existingBboxesList.push(bboxesList[i]);
                }
                dictionary.set(image_name, existingBboxesList);
            }
            else {
                dictionary.set(image_name, bboxesList);
            }
        }
        return dictionary;

        } catch(err) {
        console.error(err);
    };
}


async function displayImage(imageRef, bboxes, targetElement) {
    // Create canvas to draw on image and then bboxes
    const canvas = document.createElement("canvas");
    targetElement.appendChild(canvas);

    // Load the image
    const image = await loadImage(imageRef);

    // Adjust the canvas size to the image size
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Draw boxes
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    drawBBoxes(bboxes, ctx);
}


function loadImage(imageFile) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            URL.revokeObjectURL(this.src);
            resolve(img);
        }
        img.onerror = () => {
            reject(new Error("Failed to load image"));
        };
        img.src = URL.createObjectURL(imageFile);filteredImages.length
    });
}


function drawBBoxes(bboxes, ctx) {
    ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'red';
    for (let i = 0; i < bboxes.length; i++) {
        const [xmin, ymin, xmax, ymax] = bboxes[i];
        ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
    ctx.stroke();
}


async function displayImagesPage(lowerIndex, upperIndex, listElement) {
    for (let i = lowerIndex; i < upperIndex; i++) {
        const bboxes = imageToBBoxesMap.get(filteredImages[i].name);
        const listItemReference = document.createElement("li");
        listElement.appendChild(listItemReference);
        listItemReference.innerHTML = filteredImages[i].name + " (bbboxes_n: " + bboxes.length + ") " + i + "/" + filteredImages.length;
        await displayImage(filteredImages[i], bboxes, listItemReference);
    }
}


async function load() {
    // Parse CSV labels file
    const fileSelector = document.getElementById('label-file-selector');
    csvUrl = URL.createObjectURL(fileSelector.files[0]);
    imageToBBoxesMap = await parseBBoxCSV(csvUrl);

    // for a file.name in imageselector files display image
    const imageList = document.getElementById('image-list');
    const imageFileSelector = document.getElementById('image-file-selector');

    // Filter images that are not in the labels file
    for (let i = 0; i < imageFileSelector.files.length; i++) {
        if (imageToBBoxesMap.has(imageFileSelector.files[i].name)) {
            filteredImages.push(imageFileSelector.files[i]);
        }
    }
    filteredImages.sort((a, b) => a.name.localeCompare(b.name));

    // Display images
    displayImagesPage(0, IMG_PER_PAGE, imageList);
}

async function nextPage() {
    const imageList = document.getElementById('image-list');
    imageList.innerHTML = "";
    currentLowerIndex = (currentLowerIndex + IMG_PER_PAGE) % filteredImages.length;
    currentUpperIndex = (currentUpperIndex + IMG_PER_PAGE) % filteredImages.length;
    await displayImagesPage(currentLowerIndex, currentUpperIndex, imageList);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function previousPage() {
    const imageList = document.getElementById('image-list');
    imageList.innerHTML = "";
    currentLowerIndex = (currentLowerIndex - IMG_PER_PAGE + filteredImages.length) % filteredImages.length;
    currentUpperIndex = (currentUpperIndex - IMG_PER_PAGE + filteredImages.length) % filteredImages.length;
    await displayImagesPage(currentLowerIndex, currentUpperIndex, imageList);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Register event listeners
const loadButton = document.getElementById('load-button');
const nextPageButton = document.getElementById('next-page-button');
const previousPageButton = document.getElementById('previous-page-button');
loadButton.addEventListener("click", load, false);
nextPageButton.addEventListener("click", nextPage, false);
previousPageButton.addEventListener("click", previousPage, false);
document.getElementById('next-page-button-bot').addEventListener("click", nextPage, false);
document.getElementById('previous-page-button-bot').addEventListener("click", previousPage, false);

function concatPathWithFilename(path, filename) {
    if (path.endsWith('/')) {
      return path + filename;
    } else {
      return path + '/' + filename;
    }
  }