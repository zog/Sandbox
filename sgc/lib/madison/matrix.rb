module Madison
  require 'rubycuda'
  require 'madison/comparable'
  
  
  class Dimension
    # A vectors dimension key => value
    attr_accessor :i, :j
    
    def initialize matrix, i, j
      @matrix = matrix
      @i = i
      @j = j
    end
    
    def value= value
      @matrix.values[@i*@matrix.vectors_dimension + @j] = value
    end
    
    def key= value
      @matrix.keys[@i*@matrix.vectors_dimension + @j] = value
    end
    
    def value
      @matrix.values[@i*@matrix.vectors_dimension + @j]
    end
    
    def key
      @matrix.keys[@i*@matrix.vectors_dimension + @j]
    end
    
    def inspect
      "#<Madison::Dimension (#{i}, #{j}) #{key} => #{value}>"
    end
  end
  
  class Matrix
    include SGC::Cuda
    include Madison::Comparable
    
    attr_reader :vectors_dimension
    attr_reader :count
    attr_reader :size
    attr_accessor :keys, :values

    def initialize type, vectors_count, vectors_dimension
      @last_id = 0
      @count = vectors_count
      @vectors_dimension = vectors_dimension
      @size = vectors_count * vectors_dimension
      @type = type
      @type_size = Buffer.element_size(type)
      @dimensions = Hash.new{|h, k| h[k] = {}}
      
      # the matrix used to store the vector dimensions values
      @values = Buffer.new(type, @size)
      
      # the matrix used to store the vector dimensions keys
      @keys = Buffer.new(:int, @size)
    end
    
    def inspect
      "#<Madison::Matrix #{count}x#{vectors_dimension} @last_id=#{@last_id}>"
    end
    
    def dimensions i, j
      @dimensions[i][j] ||= Dimension.new self, i, j
    end

    def << vector
      raise "Already full" unless @last_id <= @count
      (0...[vector.size, @vectors_dimension].min).each do |k| 
        dimensions(@last_id, k).value = vector.values[k]
        dimensions(@last_id, k).key = vector.keys[k].hash
      end
      @last_id += 1
      self
    end
  end
end