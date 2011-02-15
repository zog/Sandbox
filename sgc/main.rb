require 'rubycuda'
require 'benchmark'
require 'mongo'

include SGC::Cuda

class Array; def sum; inject( nil ) { |sum,x| sum ? sum+x : x }; end; end

class WopataTest
  
  DIMENSIONS = 24
  BLOCK_SIZE = 16
  CLUSTER_SIZE = 16 # 64
  
  MONGODB_NAME = "ls"
  
  def self.run method=nil, num_blocks=nil
    # INITIALIZATION
    db = Mongo::Connection.new.db(MONGODB_NAME)
    
    method ||= :parallel
    method = method.intern
    num_blocks ||= 1
    num_blocks = num_blocks.to_i
    count = num_blocks * BLOCK_SIZE # for now, we want to use multiple of BLOCK_SIZE
    type = :float
    @type_size = Buffer.element_size(type)
    integer_size = Buffer.element_size(:int)
    
    vector_size = DIMENSIONS
    
    matrix_size = count * vector_size
    # the matrix used to store the vector dimensions values
    @matrix = Buffer.new(type, matrix_size)
    @matrix_dev_1 = CudaDeviceMemory.malloc(@type_size * matrix_size)
    @matrix_dev_2 = CudaDeviceMemory.malloc(@type_size * matrix_size)
    # the matrix used to store the vector dimensions keys
    @matrix_keys = Buffer.new(:int, matrix_size)
    @matrix_keys_dev_1 = CudaDeviceMemory.malloc(integer_size * matrix_size)
    @matrix_keys_dev_2 = CudaDeviceMemory.malloc(integer_size * matrix_size)
    
    object_ids = {}
    
    @scores_size = (CLUSTER_SIZE * BLOCK_SIZE) ** 2
    @scores = Buffer.new(type, @scores_size)
    @scores_dev = CudaDeviceMemory.malloc(@type_size * @scores_size)
    
    offset_increment = CLUSTER_SIZE * BLOCK_SIZE * DIMENSIONS
    full_scores = Hash.new {|h, k| h[k] = {}}
    
    # Initialize scores to 0
    (0...@scores_size).each do |i| @scores[i] = 0.0 end
    
    path = prepare_kernel_lib
    CudaFunction.load_lib_file(path)
    CudaMemory.memcpy_htod(@scores_dev, @scores, @scores_size * @type_size)
    
    # Initialize the Matrix, Neo
    # (0...matrix_size).each do |i| matrix[i] = rand end
    (0...count).each do |i|
      v = db['data_mining.vectors'].find(:type => 'company_tags').sort("key").skip(i).first # N queries :-S
      (0...[v["dimensions"].size, DIMENSIONS].min).each do |k|
        @matrix[i*DIMENSIONS + k] = v["dimensions"].values[k]
        @matrix_keys[i*DIMENSIONS + k] = v["dimensions"].keys[k].hash
      end
      object_ids[i] = v["key"]
    end
    
    # CudaFunction.configure(numBlocks, threadsPerBlock)
    # CudaFunction.configure(Dim3.new(count, 1, 1), Dim3.new(DIMENSIONS, 1, 1))
    # CudaFunction.setup(matrix_dev_1, count)
    # 
    # f = CudaFunction.new("MatPopulate")
    # f.launch
    
    # CudaMemory.memcpy_dtoh(matrix, matrix_dev_1, type_size * matrix_size)
    
    # Display the Matrix
    (0...count).each do |i|
     (0...DIMENSIONS).each do |j|
       $stderr.print "%.3f\t" % @matrix[i*DIMENSIONS + j]
     end
     $stderr.puts "\n"
    end
    
    # (0...count).each do |i|
    #  (0...DIMENSIONS).each do |j|
    #    $stderr.print "%s\t" % matrix_keys[(i*DIMENSIONS + j)*256]
    #  end
    #  $stderr.puts "\n"
    # end
    
    #object_ids.each_pair do |k, v|
    #  print "#{k} -> #{v}\n"
    #end
      
    
    if method == :parallel
      $stderr.puts  "Let's do it the smart way"
      clusters_count = num_blocks / CLUSTER_SIZE
      leftovers_count = num_blocks - clusters_count * CLUSTER_SIZE
      $stderr.puts  "here we have #{clusters_count} clusters"
      # We compute at most CLUSTER_SIZE blocks, each containing BLOCK_SIZE vectors
      (0...clusters_count).each do |c|
        $stderr.puts  "\n> Cluster ##{c}"
        CudaMemory.memcpy_htod(@matrix_dev_1, @matrix.offset(c*offset_increment), offset_increment * @type_size)
        CudaMemory.memcpy_htod(@matrix_keys_dev_1, @matrix_keys.offset(c*offset_increment), offset_increment * integer_size)
        
        # We will not compare twice the same vectors => f(a,b) == f(b,a)
        output_scores(CLUSTER_SIZE * BLOCK_SIZE, (c - 1) * CLUSTER_SIZE * BLOCK_SIZE, c * CLUSTER_SIZE * BLOCK_SIZE, 0, 0)
        (c...clusters_count).each do |cc|
          $stderr.puts ">> with Cluster ##{cc}"
          compare_with(c, cc, CLUSTER_SIZE, CLUSTER_SIZE)
        end
        # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
        $stderr.puts ">> with the leftovers"
        compare_with(c, clusters_count, CLUSTER_SIZE, leftovers_count)
        
        
        #CudaMemory.memcpy_htod(matrix_dev_2, @matrix.offset(clusters_count * leftovers_count * BLOCK_SIZE * DIMENSIONS), leftovers_count * BLOCK_SIZE * DIMENSIONS * type_size)
        #CudaMemory.memcpy_htod(matrix_keys_dev_2, matrix_keys.offset(clusters_count * leftovers_count * BLOCK_SIZE * DIMENSIONS), leftovers_count * BLOCK_SIZE * DIMENSIONS * type_size)
        #CudaFunction.configure(Dim3.new(leftovers_count, 1, 1), Dim3.new(BLOCK_SIZE, 1, 1))
        #CudaFunction.setup(matrix_dev_1, matrix_dev_2, matrix_keys_dev_1, matrix_keys_dev_2, scores_dev, leftovers_count * BLOCK_SIZE)
        #f = CudaFunction.new("ParallelScore")
        #f.launch
        #CudaMemory.memcpy_dtoh(scores, scores_dev, scores_size * type_size)
        #$stderr.puts "#{c * CLUSTER_SIZE * BLOCK_SIZE} .. #{c * CLUSTER_SIZE * BLOCK_SIZE + CLUSTER_SIZE * BLOCK_SIZE - 1} x #{clusters_count * CLUSTER_SIZE * BLOCK_SIZE} .. #{num_blocks * BLOCK_SIZE - 1}"
        #output_scores(CLUSTER_SIZE * BLOCK_SIZE, leftovers_count * BLOCK_SIZE, c, leftovers_count * BLOCK_SIZE, scores)
      end
      $stderr.puts  "\n> The leftovers"
      c = clusters_count
      CudaMemory.memcpy_htod(@matrix_dev_1, @matrix.offset(c*offset_increment), leftovers_count * BLOCK_SIZE * DIMENSIONS)
      CudaMemory.memcpy_htod(@matrix_keys_dev_1, @matrix_keys.offset(c*offset_increment), leftovers_count * BLOCK_SIZE * DIMENSIONS)
      
      # We will not compare twice the same vectors => f(a,b) == f(b,a)
      output_scores(leftovers_count * BLOCK_SIZE, clusters_count * CLUSTER_SIZE * BLOCK_SIZE, clusters_count * CLUSTER_SIZE * BLOCK_SIZE, 0, 0)
      
      # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
      $stderr.puts ">> with the leftovers"
      compare_with(clusters_count, clusters_count, leftovers_count, leftovers_count)
      #full_scores.keys.each do |i|
      #  full_scores[i].keys.each do |j|
      #    puts "#{i}\t #{j}\t %.3f\n" % full_scores[i][j]
      #    #unless full_scores[i][j] == full_scores[j][i]
      #    #  $stderr.puts "#{i}, #{j} != #{j}, #{i} => #{full_scores[i][j]} / #{full_scores[j][i]}"
      #    #end
      #  end
      #  puts "\n"
      #end
      
    else
      $stderr.puts "Let's do it the raw way"
      (0...count).each do |i|
        (0...count).each do |j|
          _score = 0.0
          (0...DIMENSIONS).each do |k|
            key = @matrix_keys[i*DIMENSIONS + k];
            (0...DIMENSIONS).each do |l|
              if(@matrix_keys[j*DIMENSIONS + l] == key)
                _score += score(@matrix[i*DIMENSIONS + k], @matrix[j*DIMENSIONS + l])
              end
            end
          end
          puts "#{i}\t #{j}\t %.3f\n" %full_scores[i][j] = (_score) ** 0.5
        end
        puts "\n"
      end
      #p (0...BLOCK_SIZE).map{|i| scores[i]}.sum
    end
    
    #p (0...BLOCK_SIZE).map{|i| scores[i]}
  end
  
protected
  def self.prepare_kernel_lib
    if File.exists?('kernel/libkernel.so') == false || File.mtime('kernel/kernel.cu') > File.mtime('kernel/libkernel.so') || File.mtime('kernel/kernel.h') > File.mtime('kernel/libkernel.so')
        system %{cd kernel; nvcc -shared -Xcompiler -fPIC kernel.cu -o libkernel.so}
    end
    'kernel/libkernel.so'
  end
  
  def self.score q, p
    q * p
  end
  
  def self.output_scores rows, cols, offset_x, offset_y, score
    $stderr.print rows
    $stderr.print ","
    $stderr.print cols
    $stderr.print ","
    $stderr.print offset_y
    $stderr.print ","
    $stderr.print offset_y
    $stderr.print ","
    $stderr.print score.is_a?(SGC::Memory::Buffer)
    $stderr.print "\n****\n\n"
    (0... BLOCK_SIZE).each do |i|
     (0... BLOCK_SIZE).each do |j|
       $stderr.print "%.3f\t" % score[i*cols + j]
     end
     $stderr.puts "\n"
    end
    (0...rows).each do |i|
      (0...cols).each do |j|
        real_i = offset_x + i
        real_j = offset_y + j
        #full_scores[real_i][real_j] = scores[i*CLUSTER_SIZE * BLOCK_SIZE + j]
        puts "#{real_i}\t #{real_j}\t %.3f\n" % (score.is_a?(SGC::Memory::Buffer) ? score[i * cols + j] : score)
      end
    end
  end
  
  def self.compare_with(cluster, offset, current_cluster_size, size)
    CudaMemory.memcpy_htod(@matrix_dev_2, @matrix.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * DIMENSIONS), size * BLOCK_SIZE * DIMENSIONS)
    CudaMemory.memcpy_htod(@matrix_keys_dev_2, @matrix_keys.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * DIMENSIONS), size * BLOCK_SIZE * DIMENSIONS)
    
    CudaFunction.configure(Dim3.new(size, 1, 1), Dim3.new(BLOCK_SIZE, 1, 1))
    CudaFunction.setup(@matrix_dev_1, @matrix_dev_2, @matrix_keys_dev_1, @matrix_keys_dev_2, @scores_dev, size)
    f = CudaFunction.new("ParallelScore")
    f.launch
    CudaMemory.memcpy_dtoh(@scores, @scores_dev, @scores_size * @type_size)
    
    $stderr.puts "#{cluster * CLUSTER_SIZE * BLOCK_SIZE} .. #{(cluster) * BLOCK_SIZE * CLUSTER_SIZE + current_cluster_size * BLOCK_SIZE - 1} x #{offset * CLUSTER_SIZE * BLOCK_SIZE} .. #{offset * CLUSTER_SIZE * BLOCK_SIZE + size * BLOCK_SIZE - 1}"
    output_scores(size * BLOCK_SIZE, current_cluster_size * BLOCK_SIZE, cluster * CLUSTER_SIZE * BLOCK_SIZE, offset * CLUSTER_SIZE * BLOCK_SIZE, @scores)
  end
end

time = Benchmark.measure do
  if ARGV.size > 1 && ARGV[1] == 'raw'
    WopataTest.run :raw, ARGV[0]
  else
    WopataTest.run :parallel, ARGV[0]
  end
end

$stderr.puts "Runtime: #{time}"