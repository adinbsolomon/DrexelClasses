const pg = require("pg");
const express = require("express");
const app = express();

const port = 3000;
const hostname = "localhost";

app.use(express.json());
app.use(express.static("public_html"));

// imports database environment variables
const connection = require("../env.json");

// creates new connection pool
const Pool = pg.Pool;
const pool = new Pool(connection);
pool.connect().then(function () {
    console.log("Connected!");
});

// TODO comment this out after question 1
// pool.query(
//     `INSERT INTO animals(name, age, species)
//     VALUES($1, $2, $3)
//     RETURNING *`,
//     ["Spot", 4, "dog"]
// ).then(function (response) {
//     // row was successfully inserted into table
//     console.log("Inserted:");
//     console.log(response.rows);
// })
// .catch(function (error) {
//     // something went wrong when inserting the row
//     console.log(error);
// });

let validSpecies = ["cat", "dog", "turtle", "antelope"];

app.post("/animal", function (req, res) {
    let body = req.body;
    // TODO check if name is the empty string
    // or if age is NOT a valid integer
    // or if species NOT in validSpecies
    if (
        !body.hasOwnProperty("name") ||
        !body.hasOwnProperty("age") ||
        !body.hasOwnProperty("species")
    ) {
        let name = req.body.name;
        let age = req.body.age;
        let species = req.body.species;
        if (
            name === "" ||
            !Number.isInteger(age) ||
            !validSpecies.contains(species)) {
            return res.sendStatus(200);
        } else {
            return res.sendStatus(400);
        }
    }

    // TODO run query to insert body into animals
    console.log(body);
    res.send();
});

app.get("/animal", function (req, res) {
    // TODO extract species from query string
    // TODO check if species valid, send status 400 if not
    // TODO run query selecting all animals with that species
    // TODO return rows to user, or send status 400 if query fails

    // here's a sample select query (remember to parameterize it):
    // pool.query(`SELECT * FROM animals WHERE name = 'Fluffy'`)
    res.send();
});

app.listen(port, hostname, () => {
    console.log(`Listening at: http://${hostname}:${port}`);
});


/*
YOUR ANSWERS HERE

1.

exercise5a=# SELECT * FROM animals WHERE species = 'dog';
 id | name | age | species 
----+------+-----+---------
  2 | Spot |   4 | dog
(1 row)

3c.

the client-side validation can simply be overwritten by the user in developer tools, etc.

*/
