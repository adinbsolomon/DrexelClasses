let axios = require("axios");
let express = require("express");
let app = express();

// load key from JSON file
let apiFile = require("../key.json");
let apiKey = apiFile["key"];

let port = 3000;
let hostname = "localhost";

app.use(express.static("Exercise4b/app/public_html")); // Exercise4b lives with my other material so I don't need to reinstall dependencies

app.get("/feels-like", function (req, res) {
    console.log(req.body);
    let zip = req.query.zip;
    let baseUrl = "https://api.openweathermap.org/data/2.5/weather";
    let feelsLike;
    axios.get(`${baseUrl}?zip=${zip}&appid=${apiKey}`)
        .then(function (response) {
            console.log(
                `Sent GET request to api.openweathermap.org/data/2.5/weather for zip ${zip}`
            );
            // res.json(response.data);
            feelsLike = parseInt(response.data.main.feels_like);
            feelsLike = convertKelvinToFahrenheit(feelsLike);
            res.json({
                "feelsLikeFahrenheit": feelsLike
            })
        });
    console.log("Sending request...");
    //res.json(response.data); // moved this here
});

app.listen(port, hostname, () => {
    console.log(`Listening at: http://${hostname}:${port}`);
});

function convertKelvinToFahrenheit(temp) {
    return (temp - 273.15) * (9/5) + 32;
}

/*
YOUR ANSWERS HERE

1. change the zip value in the axios call's argument

2. because the data we're sending to the client is just json data, not to be rendered as HTML

3. because the .then() function is event-based and happens after the network traffic resolves

4. because request is not defined outside of the then() call, where response is the result of the API call

5. Kelvin by default

*/
