
const express = require("express");
const app = express();

const port = 3000;
const hostname = "localhost";

app.use(express.static("public_html"));

/* returns random integer in range [min, max] */
function getRandomIntegerInRange(min, max) {
	// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random#Examples
	// add 1 to max to make the range inclusive of the max value
	return Math.floor(Math.random() * ((max + 1) - min) + min)
}

app.get("/random", function (req, res) {
	console.log("Server received query string:", req.query);
	let min = parseInt(req.query.min, 10);
	let max = parseInt(req.query.max, 10);
	let randomNumber = getRandomIntegerInRange(min, max);
	res.json({"number": randomNumber});
});

app.listen(port, hostname, () => {
	console.log(`Listening at: http://${hostname}:${port}`);
});
