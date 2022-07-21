
// the require function is very similar to Python import
// this imports the code inside the Node standard library http module
// we can now call http library functions using the http variable
let http = require("http");

// declare the hostname and the port
// that we'll listen for requests at
let hostname = "127.0.0.1";
let port = 3000;

// takes an HTTP request and response object as arguments
// the request object has these properties
// https://nodejs.org/api/http.html#http_class_http_clientrequest
// the response object has these properties
// https://nodejs.org/api/http.html#http_class_http_serverresponse
function handleRequest(req, res) {
	// prints some stuff about the request
	console.log("I got a request!");
	console.log("Request URL:", req.url);
	console.log("Request headers:", req.headers)
	console.log("Request method:", req.method);
	console.log();

	// sets the response status code
	res.statusCode = 200;

	// sets an HTTP header on the response
	res.setHeader('Content-Type', 'text/html');

	// sets the body contents and sends the response to the client
	res.end("<p>Hello! <strong>I am a paragraph!</strong></p>");
}

// handleRequest will be called whenever our program
// receives an HTTP request at http://<hostname>:<port>
// if we passed a different function to http.createServer,
// that function would be called instead
let server = http.createServer(handleRequest);

// starts the server listening for requests at http://<hostname>:<port>
// the third argument is a function that's called once
// when the server starts, and never again
// we can use it for setup code, but we usually just print a string
// that says something like "Server is listening..."
server.listen(port, hostname, function() {
  console.log(`Server listening on http://${hostname}:${port}`)
});