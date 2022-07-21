
function locationIsValid(board, r, c) {
    return (
        r >= 0 &&
        r < board.length &&
        c >= 0 &&
        c < board[0].length
    );
}

function locationIsAlive(board, r, c) {
    return locationIsValid(board, r, c) && board[r][c];
}

function range(n) {
    return Array(n).keys();
}

function countAdjacentLiveCells(board, r, c) {
    const modifiers = [-1, 0, 1];
    let count = 0;
    for (let rModifier of modifiers) {
        for (let cModifier of modifiers) {
            if (rModifier === 0 && cModifier === 0) continue; // don't count itself as adjacent
            if (locationIsAlive(board, r+rModifier, c+cModifier)) count++;
        }
    }
    return count;
}

function displayBoard(board) {
    for (let row of board) {
        let rowString = "";
        for (let cell of row) {
            rowString += (+ cell);
        }
        console.log("  " + rowString);
    }
}

function stepBoard(board) {
    let newBoard = [];
    for (let r of range(board.length)) {
        let newRow = [];
        for (let c of range(board[0].length)) {
            let adjacentLiveCells = countAdjacentLiveCells(board, r, c);
            let newCell;
            if (locationIsAlive(board, r, c)) {
                newCell = (adjacentLiveCells===2 || adjacentLiveCells===3);
            } else {
                newCell = (adjacentLiveCells===3);
            }
            newRow.push(newCell);
        }
        newBoard.push(newRow);
    }
    return newBoard;
}

function test(board) {
    console.log("\nBoard Input:")
    displayBoard(board);
    console.log("Board Output:")
    displayBoard(stepBoard(board));
}

/*
let board1 = [
    [true, true, false, true],
    [false, true, false, true],
    [false, false, false, true]
];

test(board1);

let board2 = [
    [true, true, true]
]

test(board2);
*/
