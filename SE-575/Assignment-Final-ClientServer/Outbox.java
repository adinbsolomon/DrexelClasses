import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class Outbox {

	protected ObjectOutputStream writer;

	public Outbox(ObjectOutputStream writer) {
		this.writer = writer;
	}

	public void sendMessage(Serializable object) {
		try {
			this.writer.writeObject(object);
		} catch (IOException e) {
			System.out.println("Outbox could not send message");
		}
	}

}
