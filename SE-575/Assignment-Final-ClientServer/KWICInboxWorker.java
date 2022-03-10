public class KWICInboxWorker extends KWICWorker {

	protected Inbox inbox;

	public KWICInboxWorker(KWICMediator mediator, Inbox inbox) {
		super(mediator);
		this.inbox = inbox;
	}

	public void receiveMessage() {
		this.mediator.inboxHasNewMessage((KWICMessage) this.inbox.receiveMessage());
	}

}
