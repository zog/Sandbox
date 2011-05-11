module Madison
  module Comparable
    def self.included(base)
      base.send :extend, ClassMethods
      base.send :include, InstanceMethods
    end
    
    module InstanceMethods
      include SGC::Cuda
      
      BLOCK_SIZE = 16
      CLUSTER_SIZE = 16
      INTEGER_SIZE = Buffer.element_size(:int)
      
      def compare_to matrix
        raise "Dimensions must match" unless matrix.vectors_dimension == self.vectors_dimension
        
        self_num_blocks = (self.count.to_f / BLOCK_SIZE).ceil
        self_count = self_num_blocks * BLOCK_SIZE
        self_clusters_count = self_num_blocks / CLUSTER_SIZE
        self_leftovers_count = self_num_blocks - self_clusters_count * CLUSTER_SIZE
        
        other_num_blocks = (matrix.count.to_f / BLOCK_SIZE).ceil
        other_count = other_num_blocks * BLOCK_SIZE
        other_clusters_count = other_num_blocks / CLUSTER_SIZE
        other_leftovers_count = other_num_blocks - other_clusters_count * CLUSTER_SIZE
        
        @scores_size = (CLUSTER_SIZE * BLOCK_SIZE) ** 2
        @scores = Buffer.new(@type, @scores_size)
        @scores_dev = CudaDeviceMemory.malloc(@type_size * @scores_size)
        
        # Initialize scores to 0
        (0...@scores_size).each do |i| @scores[i] = 0.0 end
        
        @values_dev_1 = CudaDeviceMemory.malloc(@type_size * @size)
        @values_dev_2 = CudaDeviceMemory.malloc(@type_size * matrix.size)
        
        @keys_dev_1 = CudaDeviceMemory.malloc(INTEGER_SIZE * @size)
        @keys_dev_2 = CudaDeviceMemory.malloc(INTEGER_SIZE * matrix.size)
        
        offset_increment = CLUSTER_SIZE * BLOCK_SIZE * self.vectors_dimension
        
        path = prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        CudaMemory.memcpy_htod(@scores_dev, @scores, @scores_size * @type_size)
        
        (0...self_clusters_count).each do |c|
          self.class.log  "\n> Cluster ##{c}"
          CudaMemory.memcpy_htod(@values_dev_1, self.values.offset(c*offset_increment), offset_increment * @type_size)
          CudaMemory.memcpy_htod(@keys_dev_1, self.keys.offset(c*offset_increment), offset_increment * INTEGER_SIZE)
          
          (0...other_clusters_count).each do |cc|
            self.class.log ">> with Cluster ##{cc}"
            compare_cluster_with(matrix, c, cc, CLUSTER_SIZE, CLUSTER_SIZE)
          end
          # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
          if other_leftovers_count > 0
            self.class.log ">> with the leftovers"
            compare_cluster_with(matrix, c, other_clusters_count, CLUSTER_SIZE, other_leftovers_count)
          end
        end
        if self_leftovers_count > 0
          self.class.log  "\n> The leftovers"
          c = self_clusters_count
          CudaMemory.memcpy_htod(@values_dev_1, self.values.offset(c*offset_increment), self_leftovers_count * BLOCK_SIZE * self.vectors_dimension * @type_size)
          CudaMemory.memcpy_htod(@keys_dev_1, self.keys.offset(c*offset_increment), self_leftovers_count * BLOCK_SIZE * self.vectors_dimension * INTEGER_SIZE)
          
          (0...other_clusters_count).each do |cc|
            self.class.log ">> with Cluster ##{cc}"
            compare_cluster_with(matrix, self_clusters_count, cc, self_leftovers_count, CLUSTER_SIZE)
          end
          # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
          if other_leftovers_count > 0
            self.class.log ">> with the leftovers"
            compare_cluster_with(matrix, self_clusters_count, other_clusters_count, self_leftovers_count, other_leftovers_count)
          end
        end
      end
      
      def compare_cluster_with(matrix, cluster, offset, current_cluster_size, size)
        puts [matrix.inspect, cluster, offset, current_cluster_size, size]
        puts matrix.inspect
        puts  size * BLOCK_SIZE * self.vectors_dimension * @type_size
        CudaMemory.memcpy_htod(@values_dev_2, matrix.values.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * self.vectors_dimension), size * BLOCK_SIZE * self.vectors_dimension * @type_size)
        CudaMemory.memcpy_htod(@keys_dev_2, matrix.keys.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * self.vectors_dimension), size * BLOCK_SIZE * self.vectors_dimension * INTEGER_SIZE)
        
        CudaFunction.configure(Dim3.new(current_cluster_size, 1, 1), Dim3.new(BLOCK_SIZE, 1, 1))
        CudaFunction.setup(@values_dev_1, @values_dev_2, @keys_dev_1, @keys_dev_2, @scores_dev, size * BLOCK_SIZE)
        f = CudaFunction.new("ParallelScore")
        f.launch
        CudaMemory.memcpy_dtoh(@scores, @scores_dev, @scores_size * @type_size)
        # @scores.each do |s| puts s end
        
        $stderr.puts "#{cluster * CLUSTER_SIZE * BLOCK_SIZE} .. #{(cluster) * BLOCK_SIZE * CLUSTER_SIZE + current_cluster_size * BLOCK_SIZE - 1} x #{offset * CLUSTER_SIZE * BLOCK_SIZE} .. #{offset * CLUSTER_SIZE * BLOCK_SIZE + size * BLOCK_SIZE - 1}"
        self.class.output_scores(current_cluster_size * BLOCK_SIZE, size * BLOCK_SIZE, cluster * CLUSTER_SIZE * BLOCK_SIZE, offset * CLUSTER_SIZE * BLOCK_SIZE, @scores)
      end
      
      def prepare_kernel_lib
        kernel_dir = "#{File.dirname(__FILE__)}/kernel"
        File.open("#{kernel_dir}/kernel.h", 'w') do |f|
          f.write "#define DIMENSIONS #{self.vectors_dimension}\n"
          f.write "#define BLOCK_SIZE #{BLOCK_SIZE}\n"
          f.write "#define CLUSTER_SIZE #{CLUSTER_SIZE}\n"
        end
        system "cd #{kernel_dir}; rm libkernel.*.so;nvcc -shared -Xcompiler -fPIC kernel.cu -o libkernel.#{self.vectors_dimension}.so"
        "#{kernel_dir}/libkernel.#{self.vectors_dimension}.so"
      end
    end
    
    module ClassMethods
      
      def log message
        $stderr.puts message
      end
      
      def output_scores rows, cols, offset_x, offset_y, score
        (0...rows).each do |i|
          (0...cols).each do |j|
            real_i = offset_x + i
            real_j = offset_y + j
            puts "#{real_i}\t #{real_j}\t %.3f\n" % (score.is_a?(SGC::Memory::Buffer) ? score[i * cols + j] : score)
          end
        end
      end
    end
  end
end