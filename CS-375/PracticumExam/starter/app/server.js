let axios = require("axios");
let express = require("express");
let env = require("../env.json");
let baseUrl = env["base_api_url"];

let app = express();
let port = 3000;
let hostname = "localhost";

app.use(express.static("PracticumExam/starter/app/public"));
//app.use(express.static("public"));
app.use(express.json());

// YOU MUST USE THIS VARIABLE TO CONSTRUCT YOUR SEPTA API REQUEST or you'll fail all of our tests! e.g.
// axios.get(`${baseUrl}...`).then(...)

// list of all valid stations on the Manayunk/Norristown line
// corresponds with <option> element values in index.html
const stations = [
    "Elm St",
    "Main St",
    "Norristown TC",
    "Conshohocken",
    "Spring Mill",
    "Miquon",
    "Ivy Ridge",
    "Manayunk",
    "Wissahickon",
    "East Falls",
    "Allegheny",
    "North Broad St",
    "Temple U",
    "Jefferson Station",
    "Suburban Station",
    "30th Street Station",
    "Penn Medicine Station",
];

/* YOUR SOLUTION GOES BELOW HERE */

app.get("/next", function(req, res) {
    if (!(req.query.hasOwnProperty("origin") && req.query.hasOwnProperty("destination"))) {
        res.status(400);
        res.json({error: "Invalid origin or destination"});
        return;
    }
    let origin = req.query.origin;
    let destination = req.query.destination;
    if (origin === destination) {
        res.status(400);
        res.json({error: "Origin and destination must be different"});
        return;
    }
    // origin and destination are valid - send the axios request
    function handleResponse(response) {
        res.status(200);
        res.json({trains: response.data});
    }
    axios.get(`${baseUrl}${origin}/${destination}`).then(handleResponse, handleResponse);
});


/* END SOLUTION */

app.listen(port, hostname, () => {
    console.log(`Listening at: http://${hostname}:${port}`);
});
