import csv
import heapq
import os

class OldFileMerger:
    def __init__(self, input_files: list, output_path: str) -> None:
        self.input_files = input_files
        self.output_path = output_path
        self.heap = []
        self.last_values = [None] * len(self.input_files)
        self.max_values_length = 0  # 记录最大的values长度
        with open(self.output_path, 'w'):
            pass

    def load_csv(self) -> None:
        print(f"[FileMerger] Starting to load CSV files...")
        for idx, file in enumerate(self.input_files):
            if not os.path.exists(file):
                print(f"[FileMerger] Warning: File does not exist: {file}")
                continue
            
            file_size = os.path.getsize(file)
            print(f"[FileMerger] Loading {file} (size: {file_size} bytes)")
            
            try:
                with open(file, 'r') as f:
                    reader = csv.reader(f)
                    row_count = 0
                    for row in reader:
                        if len(row) < 2:  # 至少需要timestamp和一个值
                            print(f"[FileMerger] Skipping invalid row in {file}: {row}")
                            continue
                        
                        timestamp = float(row[0])
                        values = [float(x) for x in row[1:]]
                        
                        # 更新最大values长度
                        if len(values) > self.max_values_length:
                            self.max_values_length = len(values)
                        
                        self.heap.append([timestamp, idx, values])
                        row_count += 1
                    
                    print(f"[FileMerger] Loaded {row_count} rows from {file}")
            except Exception as e:
                print(f"[FileMerger] Error loading {file}: {e}")
                continue
        
        if self.heap:
            heapq.heapify(self.heap)
            print(f"[FileMerger] Heapify complete, total entries: {len(self.heap)}")
        else:
            print(f"[FileMerger] Warning: No valid data loaded from any file")

    def write_csv(self) -> None:
        print(f"[FileMerger] Starting to write merged CSV...")
        
        if not self.heap:
            print(f"[FileMerger] No data to write")
            return
        
        # 如果没有找到任何有效的values长度，使用默认值
        if self.max_values_length == 0:
            print(f"[FileMerger] Warning: No valid values found, using default length of 1")
            self.max_values_length = 1
        
        try:
            with open(self.output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row_count = 0
                
                while self.heap:
                    timestamp, idx, values = heapq.heappop(self.heap)
                    self.last_values[idx] = values
                    
                    # 构建输出行
                    row = [timestamp]
                    
                    for i, vals in enumerate(self.last_values):
                        if vals is not None:
                            row.extend(vals)
                        else:
                            # 使用记录的最大长度来填充0
                            row.extend([0] * self.max_values_length)
                    
                    writer.writerow(row)
                    row_count += 1
                
                print(f"[FileMerger] Successfully wrote {row_count} rows to {self.output_path}")
        except Exception as e:
            print(f"[FileMerger] Error writing CSV: {e}")
            raise

    def __call__(self) -> None:
        print(f"[FileMerger] Merging files: {self.input_files}")
        print(f"[FileMerger] Output path: {self.output_path}")
        
        try:
            self.load_csv()
            self.write_csv()
            print(f"[FileMerger] Successfully merged {len(self.input_files)} files into {self.output_path}")
        except Exception as e:
            print(f"[FileMerger] Error during merge process: {e}")
            import traceback
            traceback.print_exc()
            raise