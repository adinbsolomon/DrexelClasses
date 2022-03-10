import java.util.ArrayList;
import java.util.TreeSet;

public class Alphabetizer {

	private TreeSet<String> tree;

	public Alphabetizer() {
		this.tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
	}

	public void addToAlphabetizedSet(String string) {
		this.tree.add(string);
	}

	public ArrayList<String> getAlphabetizedLines() {
		return new ArrayList<String>(this.tree);
	}

}
