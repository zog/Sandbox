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
      DIMENSIONS = 24
      INTEGER_SIZE = Buffer.element_size(:int)
      
      def compare_to matrix
        raise "Dimensions must match" unless (dimensions = matrix.vectors_dimension) == self.vectors_dimension
        
        num_blocks = (self.count.to_f / BLOCK_SIZE).ceil
        count = num_blocks * BLOCK_SIZE
        
        @scores_size = (CLUSTER_SIZE * BLOCK_SIZE) ** 2
        @scores = Buffer.new(@type, @scores_size)
        @scores_dev = CudaDeviceMemory.malloc(@type_size * @scores_size)
        
        # Initialize scores to 0
        (0...@scores_size).each do |i| @scores[i] = 0.0 end
        
        @values_dev_1 = CudaDeviceMemory.malloc(@type_size * @size)
        @values_dev_2 = CudaDeviceMemory.malloc(@type_size * @size)
        
        @keys_dev_1 = CudaDeviceMemory.malloc(INTEGER_SIZE * @size)
        @keys_dev_2 = CudaDeviceMemory.malloc(INTEGER_SIZE * @size)
        
        offset_increment = CLUSTER_SIZE * BLOCK_SIZE * self.vectors_dimension
        
        path = self.class.prepare_kernel_lib
        CudaFunction.load_lib_file(path)
        CudaMemory.memcpy_htod(@scores_dev, @scores, @scores_size * @type_size)
        
        clusters_count = num_blocks / CLUSTER_SIZE
        leftovers_count = num_blocks - clusters_count * CLUSTER_SIZE
        self.class.log  "here we have #{clusters_count} clusters and #{leftovers_count} leftovers"
        
        (0...clusters_count).each do |c|
          self.class.log  "\n> Cluster ##{c}"
          CudaMemory.memcpy_htod(@values_dev_1, self.values.offset(c*offset_increment), offset_increment * @type_size)
          CudaMemory.memcpy_htod(@keys_dev_1, self.keys.offset(c*offset_increment), offset_increment * INTEGER_SIZE)
          
          # We will not compare twice the same vectors => f(a,b) == f(b,a)
          output_scores(CLUSTER_SIZE * BLOCK_SIZE, c * CLUSTER_SIZE * BLOCK_SIZE, c * CLUSTER_SIZE * BLOCK_SIZE, 0, 0) if c > 0
          (c...clusters_count).each do |cc|
            self.class.log ">> with Cluster ##{cc}"
            compare_cluster_with(c, cc, CLUSTER_SIZE, CLUSTER_SIZE)
          end
          # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
          if leftovers_count > 0
            self.class.log ">> with the leftovers"
            compare_cluster_with(c, clusters_count, CLUSTER_SIZE, leftovers_count)
          end
        end
        if leftovers_count > 0
          self.class.log  "\n> The leftovers"
          c = clusters_count
          CudaMemory.memcpy_htod(@values_dev_1, self.values.offset(c*offset_increment), leftovers_count * BLOCK_SIZE * DIMENSIONS * @type_size)
          CudaMemory.memcpy_htod(@keys_dev_1, self.keys.offset(c*offset_increment), leftovers_count * BLOCK_SIZE * DIMENSIONS * INTEGER_SIZE)
          
          # We will not compare twice the same vectors => f(a,b) == f(b,a)
          self.class.output_scores(leftovers_count * BLOCK_SIZE, clusters_count * CLUSTER_SIZE  * BLOCK_SIZE, clusters_count * CLUSTER_SIZE * BLOCK_SIZE, 0, 0)
          # We have to handle the leftovers => if we have 66 blocks and CLUSTER_SIZE == 64, we have to handle 2 blocks separately
          self.class.log ">> with the leftovers"
          compare_cluster_with(clusters_count, clusters_count, leftovers_count, leftovers_count)
        end
      end
      
      def compare_cluster_with(cluster, offset, current_cluster_size, size)
        CudaMemory.memcpy_htod(@values_dev_2, self.values.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * DIMENSIONS), size * BLOCK_SIZE * DIMENSIONS * @type_size)
        CudaMemory.memcpy_htod(@keys_dev_2, self.keys.offset(offset * CLUSTER_SIZE * BLOCK_SIZE * DIMENSIONS), size * BLOCK_SIZE * DIMENSIONS * INTEGER_SIZE)
        
        CudaFunction.configure(Dim3.new(size, 1, 1), Dim3.new(BLOCK_SIZE, 1, 1))
        CudaFunction.setup(@values_dev_1, @values_dev_2, @keys_dev_1, @keys_dev_2, @scores_dev, size * BLOCK_SIZE)
        f = CudaFunction.new("ParallelScore")
        f.launch
        CudaMemory.memcpy_dtoh(@scores, @scores_dev, @scores_size * @type_size)
        
        $stderr.puts "#{cluster * CLUSTER_SIZE * BLOCK_SIZE} .. #{(cluster) * BLOCK_SIZE * CLUSTER_SIZE + current_cluster_size * BLOCK_SIZE - 1} x #{offset * CLUSTER_SIZE * BLOCK_SIZE} .. #{offset * CLUSTER_SIZE * BLOCK_SIZE + size * BLOCK_SIZE - 1}"
        self.class.output_scores(current_cluster_size * BLOCK_SIZE, size * BLOCK_SIZE, cluster * CLUSTER_SIZE * BLOCK_SIZE, offset * CLUSTER_SIZE * BLOCK_SIZE, @scores)
      end
    end
    
    module ClassMethods
      def prepare_kernel_lib
        kernel_dir = "#{File.dirname(__FILE__)}/kernel"
        if File.exists?("#{kernel_dir}/libkernel.so") == false || File.mtime("#{kernel_dir}/kernel.cu") > File.mtime("#{kernel_dir}/libkernel.so") || File.mtime("#{kernel_dir}/kernel.h") > File.mtime("#{kernel_dir}/libkernel.so")
            log "updating libkernel.so"
            system "cd #{kernel_dir}; nvcc -shared -Xcompiler -fPIC kernel.cu -o libkernel.so"
        end
        "#{kernel_dir}/libkernel.so"
      end
      
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