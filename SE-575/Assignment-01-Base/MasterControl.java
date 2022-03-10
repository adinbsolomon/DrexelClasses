import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

public class MasterControl {

	private final Input reader;
	private final CircularShifter shifter;
	private final Alphabetizer sorter;
	private final Output writer;

	public MasterControl() {
		this.reader = new Input();
		this.shifter = new CircularShifter();
		this.sorter = new Alphabetizer();
		this.writer = new Output();
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		MasterControl master = new MasterControl();
		master.start();
	}

	public void start() throws FileNotFoundException, IOException {
		ArrayList<String> lines = reader.read();
		ArrayList<String> shiftedLines = shifter.shiftLines(lines);
		ArrayList<String> sortedLines = sorter.sort(shiftedLines);
		writer.write(sortedLines);
	}

}
