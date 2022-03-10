import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer {

	public ArrayList<String> sort(List<String> lines) {
		ArrayList<String> sortedLines = new ArrayList<String>();
		TreeSet<String> tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
		tree.addAll(lines);
		for (String line : tree)
			sortedLines.add(line);
		return sortedLines;
	}

}
