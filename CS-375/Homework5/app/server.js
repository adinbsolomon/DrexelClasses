let axios = require("axios");
let express = require("express");
let app = express();

// add your API key to env.json
let apiFile = require("../env.json");
let apiKey = apiFile["api_key"];
// use this URL to make requests
let baseUrl = apiFile["base_api_url"];

let port = 3000;
let hostname = "localhost";

// don't change the name of this folder
// all client-side files should go inside here
app.use(express.static(`C:\Users\adinb\Documents\Personal\Projects\CS 375\Homework5\app\public`))

app.get("/forecast", function (req, res) {

    console.log(req.query);

    axios.get(`${baseUrl}?appid=${apiKey}&zip=${req.query.zip}`).then(function(response) {
        console.log("success");
        res.status(response.status);
        if (response.data.cod != 200) {
            res.json({error: response.data.message});
        } else {
            let forecastData = [];
            for (let forecastItem of response.data.list) {
                forecastData.push({
                    date: formatDate(forecastItem.dt_txt),
                    forecast: forecastItem.weather[0].description,
                    temperature: convertKelvinToFahrenheit(forecastItem.main.temp),
                    icon: forecastItem.weather[0].icon
                });
            }
            res.json({
                forecast_data: forecastData,
                city: response.data.city.name
            });
        }
    }, function (error) {
        console.log("failure");
        if (error.response) {
            // The request was made and the server responded with a status code that falls out of the range of 2xx
            console.log(error.response.data);
            res.status(error.response.data.cod);
            res.json({"error": error.response.data.message});
        } else if (error.request) {
            // The request was made but no response was received `error.request` is an instance of XMLHttpRequest in the browser and an instance of http.ClientRequest in node.js
            console.log(error.request);
        } else {
            // Something happened in setting up the request that triggered an Error
            console.log('Error', error.message);
        }
        //console.log(error.config);
    });

})

app.listen(port, hostname, () => {
    console.log(`Listening at: http://${hostname}:${port}`);
});

function convertKelvinToFahrenheit(temp) {
    return (temp - 273.15) * (9/5) + 32;
}

function formatDate(date) {
    let formatting = {weekday:"long", day:"numeric", month:"long", year:"numeric"};
    let dateObj = new Date(Date.parse(date));
    return `${dateObj.toLocaleString("en-us", {weekday:"long"})}, ${dateObj.getDay()} ${dateObj.toLocaleString("en-us", {month:"long"})} ${dateObj.getFullYear()} @ ${dateObj.toLocaleTimeString("en-us", {timeStyle:"short"})}`;
}
