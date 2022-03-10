import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PipeList extends Pipe {

	protected List<String> list;

	public PipeList() {
		super();
		this.list = Collections.synchronizedList(new ArrayList<>());
	}

	@Override
	public void write(String string) {
		this.list.add(string);
	}

	@Override
	public String read() {
		return this.list.remove(0);
	}

	@Override
	public boolean isEmpty() {
		return this.list.isEmpty();
	}

}
