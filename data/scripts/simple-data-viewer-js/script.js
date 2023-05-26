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


function displayImage(image, bboxes, targetElement) {
    loadImage(image, targetElement);
    drawBBoxes(bboxes, targetElement);
}


function loadImage(image, targetElement) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(image);
    img.onload = () => {
        URL.revokeObjectURL(this.src);
    }
    targetElement.appendChild(img);
    const info = document.createElement("span");
    info.innerHTML = image.name + ": " + image.size + " bytes";
    targetElement.appendChild(info);
}


function drawBBoxes(bboxes, targetElement) {
    const image = targetElement.getElementsByTagName("img");
    if (image == null) {
        throw new Error("No image found in target element");
    }
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.strokeStyle = 'red';
    for (let i = 0; i < bboxes.length; i++) {
        [xmin, ymin, xmax, ymax] = bboxes[i];
        console.log("all coordinates: ", xmin, ymin, xmax, ymax);
        ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
    }
    ctx.stroke();
    targetElement.appendChild(canvas);
}

async function load() {
    // Parse CSV labels file
    csvUrl = URL.createObjectURL(fileSelector.files[0]);
    const imageToBBoxesMap = await parseBBoxCSV(csvUrl);

    // for a file.name in imageselector files display image
    const testFile = document.getElementById('image-file-selector').files[0];
    const imageList = document.getElementById('image-list');
    const listItemReference = document.createElement("li");
    const bboxes = imageToBBoxesMap.get(testFile.name);
    console.log("bboxes: ", bboxes);
    imageList.appendChild(listItemReference);
    displayImage(testFile, bboxes, listItemReference);

    //
    //for (const [key, value] of imageToBBoxesMap) {
        //const filePath = concatPathWithFilename(imagesPath, key);
        //const file = new File([filePath], key);
    //}


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