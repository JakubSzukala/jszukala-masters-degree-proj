const fileSelector = document.getElementById('label-file-selector');
const imageContainer = document.getElementById('image-container');
const loadButton = document.getElementById('load-button');
const filePathInput = document.getElementById('file-path-input');

fileSelector.addEventListener("change", (event) => {
    const fileList = event.target.files;
    for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        console.log(file.name);
    }
});


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

            bboxesList = bboxesString.split(';');

            for (let j = 0; j < bboxesList.length; j++) {
                bboxesList[j] = bboxesList[j].split(' ').map(str => parseInt(str));
            }
            dictionary.set(image_name, bboxesList);
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
        img.src = URL.createObjectURL(imageFile);
    });
}


function drawBBoxes(bboxes, ctx) {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    for (let i = 0; i < bboxes.length; i++) {
        const [xmin, ymin, xmax, ymax] = bboxes[i];
        ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
    ctx.stroke();
}

async function load() {
    // Parse CSV labels file
    csvUrl = URL.createObjectURL(fileSelector.files[0]);
    const imageToBBoxesMap = await parseBBoxCSV(csvUrl);

    // for a file.name in imageselector files display image
    const testFile = document.getElementById('image-file-selector').files[0];
    const imageList = document.getElementById('image-list');

    const imageFileSelector = document.getElementById('image-file-selector');
    for (let i = 0; i < imageFileSelector.files.length; i++) {
        const bboxes = imageToBBoxesMap.get(imageFileSelector.files[i].name);
        if (bboxes != undefined) {
            const listItemReference = document.createElement("li");
            imageList.appendChild(listItemReference);
            listItemReference.innerHTML = imageFileSelector.files[i].name;
            await displayImage(imageFileSelector.files[i], bboxes, listItemReference);
        }
    }
}



fileSelector.addEventListener("change", displayImages, false);
loadButton.addEventListener("click", load, false);

function concatPathWithFilename(path, filename) {
    if (path.endsWith('/')) {
      return path + filename;
    } else {
      return path + '/' + filename;
    }
  }

//function drawRectangleOnImage()

function displayImages() {
    if (!this.files.length) {
        imageContainer.innerHTML = "<p>No files selected!</p>";
    } else {
        imageContainer.innerHTML = "";
        const imageList = document.createElement("ul");
        imageContainer.appendChild(imageList);

        for (let i = 0; i < this.files.length; i++) {
            const li = document.createElement("li");
            imageList.appendChild(li);

            const img = document.createElement("img");
            img.src = URL.createObjectURL(this.files[i]);
            console.log("path: ", this.files[i].path);
            img.onload = () => {
                URL.revokeObjectURL(this.src);
            }
            li.appendChild(img);
            const info = document.createElement("span");
            info.innerHTML = this.files[i].name + ": " + this.files[i].size + " bytes";
            li.appendChild(info);
        }
    }
}
// https://developer.mozilla.org/en-US/docs/Web/API/File_API/Using_files_from_web_applications#example_using_object_urls_to_display_images