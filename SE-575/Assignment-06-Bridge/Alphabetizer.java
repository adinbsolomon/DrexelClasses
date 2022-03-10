import java.util.TreeSet;

public class Alphabetizer extends Filter {

	private TreeSet<String> tree;

	public Alphabetizer(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
		this.tree = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	public void filter() {
		while (this.inPipe.isNotEmptyOrIsNotClosed()) {
			if (this.inPipe.hasNext()) {
				this.addLineToSorter(this.inPipe.read());
			}
		}
		for (String line : this.tree) {
			this.outPipe.write(line);
		}
		this.outPipe.close();
		this.stop();
	}

	private void addLineToSorter(String line) {
		this.tree.add(line);
	}

}
