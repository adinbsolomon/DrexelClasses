
/*
YOUR CLIENT SIDE CODE GOES HERE
- Remember not to modify anything in index.html as you'll only be submitting this file
- This is already sourced by index.html
*/

let getButton = document.getElementById("get-button");
getButton.addEventListener("click", function() {
    clearRows();
    clearError();
    let origin = getSelectionFromSelectInput("origin-select");
    let destination = getSelectionFromSelectInput("destination-select");
    let trainDataPromise = getTrainDataPromise(origin, destination);
    let responseStatus;
    trainDataPromise.then(function (response) {
        responseStatus = response.status;
        return response.json();
    }).then(function(data) {
        console.log(data);
        if (responseStatus != 200) {
            console.log("displaying error");
            displayError(data.error);
        } else {
            console.log("adding rows");
            for (const train of data.trains) {
                addRow(train);
            }
        }
    })
});

function getSelectionFromSelectInput(elementId) {
    let element = document.getElementById(elementId);
    return element.options[element.selectedIndex].value;
}

function getTrainDataPromise(origin, destination) {
    let url = `/next?origin=${origin}&destination=${destination}`;
    console.log(`fetching url --> ${url}`);
    let trainData = null;
    return fetch(url);
}

function clearRows() {
    let table = document.getElementById("train-table");
    while (table.children.length > 0) {
        table.removeChild(table.firstChild);
    }
}

function addRow(train) {
    let table = document.getElementById("train-table");
    let newRow = document.createElement("tr");
    newRow.appendChild(createCell(train["orig_train"]));
    newRow.appendChild(createCell(train["orig_departure_time"]));
    newRow.appendChild(createCell(train["orig_arrival_time"]));
    newRow.appendChild(createCell(train["orig_delay"]));
    table.appendChild(newRow);
    if (train["orig_delay"] == "On time") {
        newRow.style.backgroundColor = "lightgreen";
    }
}

function createCell(textContent) {
    let newCell = document.createElement("td");
    newCell.textContent = textContent;
    return newCell;
}

function clearError() {
    let messageDiv = document.getElementById("message-div");
    messageDiv.textContent = "";
}

function displayError(error) {
    let messageDiv = document.getElementById("message-div");
    messageDiv.textContent = error;
}
