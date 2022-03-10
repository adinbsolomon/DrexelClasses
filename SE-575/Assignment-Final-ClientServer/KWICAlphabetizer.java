import java.util.ArrayList;
import java.util.TreeSet;

public class KWICAlphabetizer extends KWICWorker {

	private final TreeSet<String> tree;

	public KWICAlphabetizer(KWICMediator mediator) {
		super(mediator);
		this.tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
	}

	public void add(String string) {
		this.tree.add(string);
	}

	public void sortLines() {
		ArrayList<String> sortedLines = new ArrayList<String>(this.tree);
		this.mediator.sortedSorterLines(sortedLines);
	}

}
