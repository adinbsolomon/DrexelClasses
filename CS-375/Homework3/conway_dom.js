
// board setup

function tableIsInitialized() {
    let container = document.getElementById("board-container");
    return (container.children.length > 0);
}

function makeCell(alive=false) {
    let cell = document.createElement("td");
    cell.classList.add("board-space");
    if (alive) cell.classList.add("alive");
    return cell;
}

function initializeTable(rows, columns) {
    let container = document.getElementById("board-container");
    // initialize table object
    let table = document.createElement("table");
    table.classList.add("board");
    container.appendChild(table);
    // initialize all of the rows
    for (let r of range(rows)) {
        let row = document.createElement("tr");
        for (let c of range(columns)) {
            let cell = makeCell(false);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
}

function getInitialBoard(rows, columns) {
    let board = [];
    for (let r of range(rows)) {
        let row = [];
        for (let c of range(columns)) {
            row.push((r+c)%2 === 0);
        }
        board.push(row);
    }
    return board;
}

function getRandomBool() {
    return Math.random() < 0.5;
}

function getRandomBoard(rows, columns) {
    let board = [];
    for (let r of range(rows)) {
        let row = [];
        for (let c of range(columns)) {
            row.push(getRandomBool());
        }
        board.push(row);
    }
    return board;
}

// board management

function getDimensions(board=null) {
    if (board === null) {
        let table = document.getElementsByTagName("table")[0]
        return [table.children.length, table.firstChild.children.length]
    } else {
        return [board.length, board[0].length];
    }
}

function dimensionsAgree(board) {
    let [tableRowCount, tableColCount] = getDimensions();
    let [boardRowCount, boardColCount] = getDimensions(board);
    return (
        tableRowCount === boardRowCount &&
        tableColCount === boardColCount
    );
}

function isCellAlive(cell) {
    return cell.classList.contains("alive");
}

function setCellStatus(cell, alive) {
    if (alive != isCellAlive(cell)) cell.classList.toggle("alive");
}

function getTableState() {
    let table = document.getElementsByTagName("table")[0]
    let board = [];
    for (let tableRow of table.children) {
        let boardRow = [];
        for (let tableCell of tableRow.children) {
            boardRow.push(isCellAlive(tableCell));
        }
        board.push(boardRow);
    }
    return board;
}

function updateTable(board) {
    if (!tableIsInitialized()) return;
    if (!dimensionsAgree(board)) return;
    let table = document.getElementsByTagName("table")[0]
    let [tableRowCount, tableColCount] = getDimensions();
    for (let r of range(tableRowCount)) {
        let tableRow = table.children[r];
        for (let c of range(tableColCount)) {
            let tableCell = tableRow.children[c];
            setCellStatus(tableCell, board[r][c]);
        }
    }
}

function stepTable() {
    updateTable(stepBoard(getTableState()));
}

// Interval business

let simulationInterval = null;

function startSimulation() {
    if (simulationInterval != null) return;
    simulationInterval = setInterval(stepTable, 250);
}

function stopSimulation() {
    if (simulationInterval === null) return;
    clearInterval(simulationInterval);
    simulationInterval = null;
}

// button setup

function createButton(text, onclick) {
    let container = document.getElementById("buttons-container");
    let button = document.createElement("button");
    button.textContent = text;
    button.addEventListener("click", onclick);
    container.appendChild(button);
}

function initializeButtons(boardSize) {
    createButton("Step", stepTable);
    createButton("Go", startSimulation);
    createButton("Pause", stopSimulation);
    createButton("Reset", function() {
        stopSimulation();
        updateTable(getInitialBoard(...boardSize));
    })
    createButton("Random", function() {
        stopSimulation();
        updateTable(getRandomBoard(...boardSize));
    })
}

// Initialization

console.log("Initializing the DOM");

let boardSize = [25, 25];

initializeTable(...boardSize);
updateTable(getInitialBoard(...boardSize));

initializeButtons(boardSize);
