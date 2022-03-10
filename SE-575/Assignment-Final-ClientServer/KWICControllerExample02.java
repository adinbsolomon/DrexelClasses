import java.util.ArrayList;

public class KWICControllerExample02 {

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

		// Initialize the clients on their own threads
		KWICControllerClient controllerClientA = new KWICControllerClient("Client A", getClientAStrings());
		KWICControllerClient controllerClientB = new KWICControllerClient("Client B", getClientBStrings());
		KWICControllerClient controllerClientC = new KWICControllerClient("Client C", getClientCStrings());

		// start the client threads
		controllerClientA.start();
		controllerClientB.start();
		controllerClientC.start();

	}

	public static ArrayList<String> getClientAStrings() {
		ArrayList<String> strings = new ArrayList<>();
		strings.add("ClientA's first line");
		strings.add("ClientA's second line");
		return strings;
	}

	public static ArrayList<String> getClientBStrings() {
		ArrayList<String> strings = new ArrayList<>();
		strings.add("ClientB's first line");
		strings.add("ClientB's second line");
		return strings;
	}

	public static ArrayList<String> getClientCStrings() {
		ArrayList<String> strings = new ArrayList<>();
		strings.add("ClientC's first line");
		strings.add("ClientC's second line");
		return strings;
	}

}
