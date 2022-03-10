import java.util.ArrayList;
import java.util.TreeSet;

public class Alphabetizer extends Subject implements Observer {

	private TreeSet<String> tree;

	public Alphabetizer() {
		this.tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	public void update(Message msg) {
		if (msg.getBool()) {
			for (String line : this.getAlphabetizedLines()) {
				notifyAllObservers(new Message(line));
			}
			this.notifyAllObservers(Message.finished);
		} else {
			this.tree.add(msg.getString());
		}
	}

	public ArrayList<String> getAlphabetizedLines() {
		return new ArrayList<String>(this.tree);
	}

}
