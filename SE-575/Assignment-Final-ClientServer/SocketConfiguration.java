import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

public abstract class SocketConfiguration {

	protected String host;
	protected int port;

	public SocketConfiguration(String host, int port) {
		this.host = host;
		this.port = port;
	}

	public static Inbox getInboxFromSocket(Socket socket) {
		try {
			InputStream is = socket.getInputStream();
			ObjectInputStream ois = new ObjectInputStream(is);
			return new Inbox(ois);
		} catch (IOException e) {
			System.out.println("Inbox creation failed");
		}
		return null;
	}

	public static Outbox getOutboxFromSocket(Socket socket) {
		try {
			OutputStream os = socket.getOutputStream();
			ObjectOutputStream oos = new ObjectOutputStream(os);
			return new Outbox(oos);
		} catch (IOException e) {
			System.out.println("Outbox creation failed");
		}
		return null;
	}

	public ServerSocket getServerSocket() {
		try {
			return new ServerSocket(this.port);
		} catch (IOException e) {
			System.out.println("Failed to create ServerSocket");
		}
		return null;
	}

	public Socket getClientSocket() {
		try {
			return new Socket(this.host, this.port);
		} catch (IOException e) {
			System.out.println("Failed to create the client's Socket");
		}
		return null;
	}

	public String getHost() {
		return this.host;
	}

	public int getPort() {
		return this.port;
	}

}
