import java.net.Socket;
import java.util.ArrayList;

public class KWICMediator implements Runnable {

	protected KWICInboxWorker inbox;
	protected KWICCircularShifter shifter;
	protected KWICAlphabetizer sorter;
	protected Outbox outbox;
	protected boolean done;

	public KWICMediator(Socket clientSocket) {

		this.outbox = SocketConfiguration.getOutboxFromSocket(clientSocket);

		Inbox basicInbox = SocketConfiguration.getInboxFromSocket(clientSocket);
		this.inbox = new KWICInboxWorker(this, basicInbox);

		this.shifter = new KWICCircularShifter(this);
		this.sorter = new KWICAlphabetizer(this);

	}

	@Override
	public void run() {
		while (!this.done) {
			this.inbox.receiveMessage();
		}
	}

	public void inboxHasNewMessage(KWICMessage message) {
		boolean messageIsDone = message.getBool();
		if (messageIsDone) {
			// stop the thread from looping
			this.done = true;
			// sort accumulated lines
			this.sorter.sortLines();
		} else {
			String string = message.getString();
			this.shifter.shiftLine(string);
		}
	}

	public void shifterHasNewLine(String line) {
		this.sorter.add(line);
	}

	public void sortedSorterLines(ArrayList<String> sortedLines) {
		// send sorted lines back to client
		for (String line : sortedLines) {
			this.outbox.sendMessage(new KWICMessage(line));
		}
		// tell client that the sorted lines have all been sent
		this.outbox.sendMessage(new KWICMessage(true));
	}

}
