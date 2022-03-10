import java.io.IOException;
import java.io.ObjectInputStream;

public class Inbox {

	protected ObjectInputStream reader;

	public Inbox(ObjectInputStream reader) {
		this.reader = reader;
	}

	public Message receiveMessage() {
		try {
			return (Message) this.reader.readObject();
		} catch (IOException | ClassNotFoundException e) {
			System.out.println("Inbox could not get message");
		}
		return null;
	}

}
