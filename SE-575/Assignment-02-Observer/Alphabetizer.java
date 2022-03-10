import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer implements Observer {

	private TreeSet<String> tree;

	public Alphabetizer() {
		this.tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	public void update(String string) {
		this.tree.add(string);
	}

	public ArrayList<String> getAlphabetizedLines() {
		return new ArrayList<String>(this.tree);
	}

}
