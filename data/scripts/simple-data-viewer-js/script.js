const fileSelector = document.getElementById('file-selector');
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


async function load() {
    // Parse CSV labels file
    csvUrl = URL.createObjectURL(fileSelector.files[0]);
    const imageToBBoxesMap = await parseBBoxCSV(csvUrl);
    const imagesPath = filePathInput.value;

    // Create container for images to display
    const imageList = document.createElement("ul");
    imageContainer.appendChild(imageList);
    const testFileName = 'e6b6a900e5c54cd5d8b0649768c361512cff1813409319eba26da5c7f47bb2e6.png'
    const testFile = new File([imagesPath], testFileName);
    // 
    //for (const [key, value] of imageToBBoxesMap) {
        //const filePath = concatPathWithFilename(imagesPath, key);
        //const file = new File([filePath], key);
    //}


}


//fileSelector.addEventListener("change", displayImages, false);
loadButton.addEventListener("click", load, false);

function concatPathWithFilename(path, filename) {
    if (path.endsWith('/')) {
      return path + filename;
    } else {
      return path + '/' + filename;
    }
  }

//function drawRectangleOnImage()

function displayImage(image) {

}

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