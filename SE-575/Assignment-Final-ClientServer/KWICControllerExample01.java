import java.util.ArrayList;

public class KWICControllerExample01 {

	public static void main(String[] args) {

		// Initialize a server on its own thread
		KWICControllerServer controllerServer = new KWICControllerServer();

		// Start up the server thread
		controllerServer.start();

		// Let the server get up to speed before trying to connect
		try {
			Thread.sleep(1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		// Initialize the client on its own thread
		KWICControllerClient controllerClient = new KWICControllerClient("The only client", getClientStrings());

		// start the client thread
		controllerClient.start();

	}

	public static ArrayList<String> getClientStrings() {
		ArrayList<String> strings = new ArrayList<>();
		strings.add("This is the first test");
		strings.add("This is the second test");
		strings.add("This seems like the last test");
		return strings;
	}

}
