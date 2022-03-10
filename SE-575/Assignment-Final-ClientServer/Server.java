import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;

public abstract class Server {

	protected SocketConfiguration socketConfiguration;
	protected ServerSocket serverSocket;
	protected ArrayList<Thread> clientThreads;

	public Server(SocketConfiguration socketConfiguration) {
		this.socketConfiguration = socketConfiguration;
		this.serverSocket = this.socketConfiguration.getServerSocket();
		this.clientThreads = new ArrayList<>();
	}

	public void serve() {
		while (true) {
			try {
				Socket newClientSocket = this.serverSocket.accept();
				Thread clientThread = this.handleClient(newClientSocket);
				this.clientThreads.add(clientThread);
				clientThread.start();
			} catch (IOException e) {
				System.out.println("Server failed to receive a connection");
				break;
			}
		}
	}

	public abstract Thread handleClient(Socket clientSocket);

}
