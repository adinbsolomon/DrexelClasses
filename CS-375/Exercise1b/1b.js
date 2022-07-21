
console.log("\nPart 1\n");

for (let i of Array(51).keys()) {
    console.log(i + 50);
}

console.log("\nPart 2\n");

let myObj = {
    a: 1,
    anotherProp: "applesauce",
    returnOne: function() {return 1;}
}

console.log(myObj.returnOne());

console.log("\nPart 3\n");

function print2d(a) {
    for (let outer of a) {
        for (let inner of outer) {
            console.log(inner);
        }
    }
}

array2d = [
    [1, 2, 3],
    ['a', 'b'],
    [77]
];

print2d(array2d);

console.log("\nPart 4a\n");

function dimensions(a) {
    return [a.length, a[0].length];
}

array2d = [
    [1, 2, 3],
    [4, 5, 6]
];

console.log(dimensions(array2d));

console.log("\nPart 4b\n");

function isUpperLeftValid(a, r, c) {
    let dims = dimensions(a);
    return (
        r-1 >= 0 &&
        r-1 < dims[0] &&
        c-1 >= 0 &&
        c-1 < dims[1]
    );
}

console.log(isUpperLeftValid(array2d, 1, 1));

console.log("\nPart 4c\n");

function isPositionValid(a, r, c) {
    return (
        r >= 0 &&
        r < a.length &&
        c >= 0 &&
        c < a[0].length
    );
}

modifiers = [-1, 1];
function sumDiagonals(a, r, c) {
    let rModifier, cModifier, sum = 0;
    for (rModifier of modifiers) {
        for (cModifier of modifiers) {
            if (isPositionValid(a, r+rModifier, c+cModifier)) sum += a[r+rModifier][c+cModifier];
        }
    }
    return sum;
}

array2d = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
]

console.log(sumDiagonals(array2d, 99, 99));
console.log(sumDiagonals(array2d, 1, 2));
console.log(sumDiagonals(array2d, 2, 0));
console.log(sumDiagonals(array2d, 1, 1));