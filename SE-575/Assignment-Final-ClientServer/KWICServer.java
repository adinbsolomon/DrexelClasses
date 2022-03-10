import java.net.Socket;

public class KWICServer extends Server {

	public KWICServer() {
		super(new KWICSocketConfiguration());
	}

	@Override
	public Thread handleClient(Socket clientSocket) {

		KWICMediator mediator = new KWICMediator(clientSocket);
		return new Thread(mediator);

	}

}
