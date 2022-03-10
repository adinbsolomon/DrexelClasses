public class KWICControllerServer extends Thread {

	@Override
	public void run() {

		// Initialize the server
		KWICServer server = new KWICServer();

		// Start the server's serving!
		server.serve();

	}

}
