import java.util.ArrayList;

public class KWICControllerClient extends Thread {

	protected String clientID;
	protected ArrayList<String> strings;

	public KWICControllerClient(String clientID, ArrayList<String> strings) {
		this.clientID = clientID;
		this.strings = strings;
	}

	@Override
	public void run() {

		// Initialize the client
		KWICClient client = new KWICClient(this.clientID);

		// Have the client send messages to the server
		for (String s : this.strings) {
			client.sendString(s);
		}

		// Client lets server know when it's done
		client.sendMessage(new KWICMessage(true));

	}

}
