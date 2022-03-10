import java.io.IOException;

public class MasterControl {

	private Input reader;
	private CircularShifter shifter;
	private Alphabetizer sorter;
	private Output writer;

	public static void main(String[] args) throws IOException {
		MasterControl master = new MasterControl();
		master.start();
	}

	public void start() throws IOException {
		// Instantiate components here
		this.reader = new Input();
		this.shifter = new CircularShifter();
		this.sorter = new Alphabetizer();
		this.writer = new Output();

		this.reader.attach(this.shifter);
		this.shifter.attach(this.sorter);
		this.sorter.attach(this.writer);

		this.reader.read();

		// Garbage collect components in case start() is called more than once
		this.reader = null;
		this.shifter = null;
		this.sorter = null;
		this.writer = null;
	}

}
