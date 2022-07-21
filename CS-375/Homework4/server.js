
let express = require("express");
let app = express();
app.use(express.json(strict=false));

let port = 3000;
let hostname = "localhost";

let readings = {}; // deviceID : [reading1, reading2, ...]
function haveNoReadingsFrom(deviceID) {
    return !(deviceID in readings);
}
function getReadingsFrom(deviceID) {
    return readings[deviceID];
}
function addReading(deviceID, reading) {
    if (haveNoReadingsFrom(deviceID)) {
        readings[deviceID] = [reading];
    } else {
        readings[deviceID].push(reading);
    }
}
function displayReadings(deviceID=null) {
    if (deviceID === null) {
        console.log(readings);
    } else {
        if (haveNoReadingsFrom(deviceID)) {
            console.log([]);
        } else {
            console.log(readings[deviceID]);
        }
    }
}

function sendResponseInvalidRequest(res, log="") {
    console.log("Invalid Request... " + log);
    res.setHeader('Content-Type', 'text/plain')
    res.status(400);
    res.send('Invalid Request');
}
function sendResponseDeviceNotFound(res, log="") {
    console.log("Device Not Found... " + log);
    res.setHeader('Content-Type', 'text/plain');
    res.status(404);
    res.send("Device Not Found");
}
function getHTML() {
    function makeRow(deviceID) {
        let deviceReadings = [...readings[deviceID]];
        let styling;
        if (deviceReadings.reduce(function(a,b){return a+b;}) > 1000) {
            styling = "\"background-color: red\"";
        } else {
            styling = "";
        }
        let deviceReadingsString = "" + deviceReadings[0];
        deviceReadings.shift();
        for (num of deviceReadings) {
            deviceReadingsString += ", " + num;
        }
        return `
            <tr style=${styling}>
                <td>${deviceID}</td>
                <td>${deviceReadingsString}</td>
            </tr>
        `;
    }
    function makeRows() {
        tableRows = "";
        for (let deviceID in readings) {
            tableRows += makeRow(deviceID);
        }
        return tableRows;
    }
    tableRows = makeRows();
    return `<!DOCTYPE html>
    <html lang="en">
    
    <head>
      <meta />
      <meta charset="utf-8">
      <title>Change_me</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <meta name="author" content="">
      <script src="http://code.jquery.com/jquery-latest.min.js"></script>
    </head>
    
    <body>
      
    <table style="border: 1px solid black">
        <tbody>
            <tr>
                <th>Device ID</th>
                <th>Energy Usage</th>
            <tr>
            ${tableRows}
        </tbody>
    </table>
    
    </body>
    </html>`;
}

app.get("/", function (req, res) {
    console.log("base");
    res.send(getHTML());
})

app.post("/api/:deviceID", function(req, res) {
    console.log("post --> " + req.params.deviceID);
    if (req.get('Content-Type') != 'application/json') {
        sendResponseInvalidRequest(res, log="content-type");
    } else {
        if (!req.body.hasOwnProperty('energy-usage')) {
            sendResponseInvalidRequest(res, log="req.body");
        } else {
            let deviceID = req.params.deviceID;
            let energyUsage = req.body['energy-usage'];
            console.log(deviceID, energyUsage);
            if (!Number.isInteger(energyUsage)) {
                sendResponseInvalidRequest(res, log="invalid energy-usage: "+energyUsage)
            } else {
                addReading(deviceID, energyUsage);
                displayReadings();
            }
        }
    }
    res.status(200);
    res.send();
})

app.get("/api/:deviceID", function (req, res) {
    console.log("get --> " + req.params.deviceID);
    let deviceID = req.params.deviceID;
    if (haveNoReadingsFrom(deviceID)) {
        sendResponseDeviceNotFound(res, log=deviceID);
    } else {
        res.status(200);
        responseJSON = {
            "total-energy-usage": getReadingsFrom(deviceID)
        };
        console.log(responseJSON);
        res.json(responseJSON);
    }
});

app.listen(port, hostname, () => {
    console.log(`Listening at: http://${hostname}:${port}`);
});
