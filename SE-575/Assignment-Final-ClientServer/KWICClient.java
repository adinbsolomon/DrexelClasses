public class KWICClient extends Client {

	public static boolean LOGGING = true;

	protected String id;

	public KWICClient(String id) {
		super(new KWICSocketConfiguration());
		this.id = id;
		this.log("Connected to the server");
	}

	public void sendString(String string) {
		this.sendMessage(new KWICMessage(string));
	}

	@Override
	public void sendMessage(Message message) {
		this.outbox.sendMessage(message);
		boolean thisMessageIsAFinalMessage = message.getBool();
		if (thisMessageIsAFinalMessage) {
			this.log("Sent final message - waiting for incoming messages");
			while (true) {
				KWICMessage incomingMessage = this.receiveMessage();
				boolean noMoreMessages = incomingMessage.getBool();
				if (noMoreMessages) {
					break;
				} else {
					if (LOGGING) {
						this.log("Received message: " + incomingMessage.getString());
					} else {
						System.out.println(incomingMessage.getString());
					}
				}
			}
		} else {
			this.log("Sent a message: " + message.getString());
		}
	}

	@Override
	public KWICMessage receiveMessage() {
		KWICMessage incomingMessage = (KWICMessage) this.inbox.receiveMessage();
		return incomingMessage;
	}

	protected void log(String string) {
		if (LOGGING) {
			System.out.println(this.id + " --> " + string);
		}
	}

}
