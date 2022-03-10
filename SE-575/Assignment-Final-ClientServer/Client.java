import java.net.Socket;

public abstract class Client {

	protected SocketConfiguration socketConfiguration;
	protected Socket socket;
	protected Inbox inbox;
	protected Outbox outbox;

	public Client(SocketConfiguration socketConfiguration) {
		this.socketConfiguration = socketConfiguration;
		this.socket = this.socketConfiguration.getClientSocket();
		this.outbox = SocketConfiguration.getOutboxFromSocket(this.socket);
		this.inbox = SocketConfiguration.getInboxFromSocket(this.socket);
	}

	public abstract void sendMessage(Message message);

	public abstract Message receiveMessage();

}
